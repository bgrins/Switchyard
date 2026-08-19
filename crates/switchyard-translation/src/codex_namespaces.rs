// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Codex tool namespace preservation across a Responses/Chat translation.
//!
//! Codex groups tools into non-standard ``namespace`` containers — MCP servers,
//! usually behind an `mcp__` prefix, plus builtin groups such as
//! `multi_agent_v1` — and dispatches on the pair of tool name and namespace. It
//! therefore expects the namespace back on the call it receives:
//! `{"type": "function_call", "name": "search", "namespace": "mcp__docs"}`.
//!
//! OpenAI-compatible upstreams accept only flat `function` tools, so the request
//! codec flattens the containers. Each child keeps its namespace on the
//! [`ToolDefinition`], and the target encoder qualifies the wire name as
//! `<namespace>__<tool>`. Two tools that differ only by namespace therefore stay
//! distinct upstream, and this module maps the returned name back to the Codex
//! name and namespace.
//!
//! No container is *filtered* by its name, because Codex resolves a call with no
//! namespace against its default group: dropping a builtin group's namespace
//! would look the call up in the wrong place. The `mcp__` prefix appears only in
//! one fallback spelling, never as a condition for handling a namespace.

use std::collections::{HashMap, HashSet};

use serde_json::Value;
use switchyard_protocol::ToolDefinition;

/// Separator between a namespace and a tool name in a qualified wire name.
pub const NAMESPACE_SEPARATOR: &str = "__";

/// Returns the wire name for a tool, qualified by its namespace when it has one.
pub fn qualified_tool_name(tool: &ToolDefinition) -> String {
    match tool.namespace() {
        Some(namespace) => format!("{namespace}{NAMESPACE_SEPARATOR}{}", tool.name),
        None => tool.name.clone(),
    }
}

/// Returns the wire name for a recorded call, qualified when it has a namespace.
///
/// History has to spell a tool the same way its definition does, or the
/// transcript teaches the model a name the upstream never offered.
pub fn qualified_call_name(call: &switchyard_protocol::ToolCall) -> String {
    match call.namespace() {
        Some(namespace) => format!("{namespace}{NAMESPACE_SEPARATOR}{}", call.name),
        None => call.name.clone(),
    }
}

/// Reverse map from an upstream tool name to its Codex name and namespace.
///
/// Built from the decoded request's tools, which carry the namespace they were
/// grouped under. The exact qualified name is always registered. A model often
/// returns a near miss instead, so two fallback spellings are registered when
/// neither can be confused with another tool:
///
/// * the qualified name without the `mcp__` prefix
/// * the bare tool name, only when exactly one namespaced tool claims it and no
///   un-namespaced tool shares it
///
/// A name absent from the map is left alone, so an unrecognized call reaches
/// Codex unchanged rather than being attributed to the wrong namespace.
pub fn qualified_tool_origins(tools: &[ToolDefinition]) -> HashMap<String, (String, String)> {
    let mut namespaced_claims: HashMap<&str, usize> = HashMap::new();
    let mut unnamespaced: HashSet<&str> = HashSet::new();
    for tool in tools {
        match tool.namespace() {
            Some(_) => *namespaced_claims.entry(tool.name.as_str()).or_default() += 1,
            None => {
                unnamespaced.insert(tool.name.as_str());
            }
        }
    }

    let mut origins = HashMap::new();
    for tool in tools {
        let Some(namespace) = tool.namespace() else {
            continue;
        };
        let origin = (tool.name.clone(), namespace.to_string());
        if let Some(stripped) = namespace.strip_prefix("mcp__") {
            let spelling = format!("{stripped}{NAMESPACE_SEPARATOR}{}", tool.name);
            if !unnamespaced.contains(spelling.as_str()) {
                origins.entry(spelling).or_insert_with(|| origin.clone());
            }
        }
        if namespaced_claims.get(tool.name.as_str()) == Some(&1)
            && !unnamespaced.contains(tool.name.as_str())
        {
            origins
                .entry(tool.name.clone())
                .or_insert_with(|| origin.clone());
        }
        // The exact spelling always wins over a fallback.
        origins.insert(qualified_tool_name(tool), origin);
    }
    origins
}

/// Rewrite `function_call` names back to the Codex name plus namespace.
///
/// Walks the whole value, covering a buffered body and each streaming event,
/// where the item is nested under `item` (`response.output_item.added` /
/// `.done`) or `response.output` (`response.completed`). An existing
/// `namespace` is never overwritten.
pub fn restore_qualified_tool_names(body: &mut Value, origins: &HashMap<String, (String, String)>) {
    if origins.is_empty() {
        return;
    }
    match body {
        Value::Array(values) => {
            for value in values {
                restore_qualified_tool_names(value, origins);
            }
        }
        Value::Object(object) => {
            if object.get("type").and_then(Value::as_str) == Some("function_call")
                && let Some(name) = object.get("name").and_then(Value::as_str)
                && let Some((tool, namespace)) = origins.get(name)
            {
                object.insert("name".to_string(), Value::String(tool.clone()));
                object
                    .entry("namespace".to_string())
                    .or_insert_with(|| Value::String(namespace.clone()));
            }
            for value in object.values_mut() {
                restore_qualified_tool_names(value, origins);
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{
        ToolDefinition, qualified_tool_name, qualified_tool_origins, restore_qualified_tool_names,
    };

    fn tool(name: &str, namespace: Option<&str>) -> ToolDefinition {
        let mut tool = ToolDefinition {
            name: name.to_string(),
            ..Default::default()
        };
        if let Some(namespace) = namespace {
            tool.set_namespace(namespace);
        }
        tool
    }

    #[test]
    fn qualifies_only_namespaced_tools() {
        assert_eq!(
            qualified_tool_name(&tool("search", Some("mcp__docs"))),
            "mcp__docs__search"
        );
        assert_eq!(qualified_tool_name(&tool("shell", None)), "shell");
    }

    // The point of qualifying: two servers exposing one name stay distinct
    // upstream, and each call resolves back to the server it came from.
    #[test]
    fn resolves_the_same_tool_name_under_two_namespaces() {
        let origins = qualified_tool_origins(&[
            tool("search", Some("mcp__a")),
            tool("search", Some("mcp__b")),
        ]);
        let mut response = json!({
            "output": [{"type": "function_call", "name": "mcp__b__search", "arguments": "{}"}]
        });

        restore_qualified_tool_names(&mut response, &origins);

        assert_eq!(response["output"][0]["name"], "search");
        assert_eq!(response["output"][0]["namespace"], "mcp__b");
    }

    // Models drop the `mcp__` prefix, so that spelling resolves too.
    #[test]
    fn resolves_a_name_missing_the_mcp_prefix() {
        let origins = qualified_tool_origins(&[tool("get_secret_word", Some("mcp__secret"))]);
        let mut response = json!({
            "output": [{"type": "function_call", "name": "secret__get_secret_word"}]
        });

        restore_qualified_tool_names(&mut response, &origins);

        assert_eq!(response["output"][0]["name"], "get_secret_word");
        assert_eq!(response["output"][0]["namespace"], "mcp__secret");
    }

    // An unambiguous bare name resolves, so a model that drops the namespace
    // entirely still dispatches.
    #[test]
    fn resolves_an_unambiguous_bare_name() {
        let origins = qualified_tool_origins(&[tool("get_secret_word", Some("mcp__secret"))]);
        let mut response = json!({
            "output": [{"type": "function_call", "name": "get_secret_word"}]
        });

        restore_qualified_tool_names(&mut response, &origins);

        assert_eq!(response["output"][0]["namespace"], "mcp__secret");
    }

    // A bare name claimed by two namespaces, or shared with an un-namespaced
    // tool, must not be guessed: a wrong guess dispatches to the wrong server.
    #[test]
    fn leaves_an_ambiguous_bare_name_alone() {
        let two_namespaces = qualified_tool_origins(&[
            tool("search", Some("mcp__a")),
            tool("search", Some("mcp__b")),
        ]);
        let shared_with_builtin =
            qualified_tool_origins(&[tool("search", Some("mcp__fs")), tool("search", None)]);

        for origins in [two_namespaces, shared_with_builtin] {
            let mut response = json!({
                "output": [{"type": "function_call", "name": "search"}]
            });
            let before = response.clone();
            restore_qualified_tool_names(&mut response, &origins);
            assert_eq!(response, before, "a bare ambiguous name must be left alone");
        }
    }

    // Streaming events nest the item one level deeper than a buffered body.
    #[test]
    fn rewrites_nested_streaming_items() {
        let origins = qualified_tool_origins(&[tool("search", Some("mcp__docs"))]);
        let mut added = json!({
            "type": "response.output_item.added",
            "item": {"type": "function_call", "name": "mcp__docs__search", "arguments": ""}
        });
        let mut completed = json!({
            "type": "response.completed",
            "response": {
                "output": [{"type": "function_call", "name": "mcp__docs__search"}]
            }
        });

        restore_qualified_tool_names(&mut added, &origins);
        restore_qualified_tool_names(&mut completed, &origins);

        assert_eq!(added["item"]["name"], "search");
        assert_eq!(added["item"]["namespace"], "mcp__docs");
        assert_eq!(completed["response"]["output"][0]["namespace"], "mcp__docs");
    }

    // Codex namespaces builtin groups too, so nothing may key on `mcp__`.
    #[test]
    fn qualifies_namespaces_that_are_not_mcp_servers() {
        let origins = qualified_tool_origins(&[tool("spawn_agent", Some("multi_agent_v1"))]);
        let mut response = json!({
            "output": [{"type": "function_call", "name": "multi_agent_v1__spawn_agent"}]
        });

        restore_qualified_tool_names(&mut response, &origins);

        assert_eq!(response["output"][0]["name"], "spawn_agent");
        assert_eq!(response["output"][0]["namespace"], "multi_agent_v1");
    }

    // An upstream that already supplied a namespace is trusted.
    #[test]
    fn preserves_an_upstream_supplied_namespace() {
        let origins = qualified_tool_origins(&[tool("search", Some("mcp__docs"))]);
        let mut response = json!({
            "output": [{
                "type": "function_call",
                "name": "mcp__docs__search",
                "namespace": "mcp__upstream"
            }]
        });

        restore_qualified_tool_names(&mut response, &origins);

        assert_eq!(response["output"][0]["namespace"], "mcp__upstream");
    }

    #[test]
    fn ignores_requests_without_namespaced_tools() {
        assert!(qualified_tool_origins(&[tool("shell", None)]).is_empty());
    }
}
