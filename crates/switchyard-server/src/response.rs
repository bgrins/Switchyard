// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Response encoding glue for libsy server endpoints.

use std::collections::HashMap;
use std::error::Error;

use axum::Json;
use axum::response::{IntoResponse, Response as HttpResponse};
use futures_util::StreamExt;
use serde_json::Value;
use switchyard_protocol::{LlmResponse, Response as AlgorithmResponse};
use switchyard_translation::{WireFormat, encode_aggregated_response, encode_stream};

use crate::sse::frame_stream;

type BoxError = Box<dyn Error + Send + Sync>;

/// Encodes a libsy response into the endpoint's wire format, reporting
/// `served_model` as the response model so the body names the model that
/// answered rather than the route the caller addressed.
pub(crate) fn into_http_response(
    response: AlgorithmResponse,
    target_format: WireFormat,
    served_model: Option<String>,
    response_namespaces: HashMap<String, String>,
) -> Result<HttpResponse, BoxError> {
    match response.llm_response {
        LlmResponse::Agg(response) => {
            let mut body =
                encode_aggregated_response(&response, target_format, served_model.as_deref())?;
            restore_responses_tool_namespaces(&mut body, &response_namespaces);
            Ok(Json(body).into_response())
        }
        LlmResponse::Stream(stream) => {
            let events = encode_stream(stream, target_format, served_model)?;
            let events = events.map(move |event| {
                event.map(|mut value| {
                    restore_responses_tool_namespaces(&mut value, &response_namespaces);
                    value
                })
            });
            Ok(frame_stream(Box::pin(events), target_format).into_response())
        }
    }
}

/// Return the MCP namespace for every unambiguous Responses function tool.
///
/// Codex wraps every MCP server's functions in its non-standard ``namespace``
/// tool container. The proxy flattens the children before sending them to a
/// Chat-only upstream; this lookup lets it put the namespace back on returned
/// Responses function calls so Codex can dispatch them to the right MCP server.
pub(crate) fn responses_tool_namespaces(
    body: &Value,
    wire_format: WireFormat,
) -> HashMap<String, String> {
    if wire_format != WireFormat::OpenAiResponses {
        return HashMap::new();
    }
    let Some(tools) = body.get("tools").and_then(Value::as_array) else {
        return HashMap::new();
    };
    let mut namespaces = HashMap::new();
    collect_responses_tool_namespaces(tools, None, &mut namespaces);
    namespaces
        .into_iter()
        .filter_map(|(name, namespace)| namespace.map(|namespace| (name, namespace)))
        .collect()
}

// A None entry marks a duplicate tool name under different MCP namespaces. A
// plain function call cannot disambiguate those safely, so it remains flat.
fn collect_responses_tool_namespaces(
    tools: &[Value],
    parent_namespace: Option<&str>,
    namespaces: &mut HashMap<String, Option<String>>,
) {
    for tool in tools {
        let Some(tool) = tool.as_object() else {
            continue;
        };
        if tool.get("type").and_then(Value::as_str) == Some("namespace") {
            let namespace = tool.get("name").and_then(Value::as_str);
            if let Some(children) = tool.get("tools").and_then(Value::as_array) {
                collect_responses_tool_namespaces(children, namespace, namespaces);
            }
            continue;
        }
        let Some(namespace) = parent_namespace else {
            continue;
        };
        let name = tool
            .get("function")
            .and_then(Value::as_object)
            .and_then(|function| function.get("name"))
            .or_else(|| tool.get("name"))
            .or_else(|| tool.get("id"))
            .and_then(Value::as_str)
            .filter(|name| !name.is_empty());
        let Some(name) = name else {
            continue;
        };
        match namespaces.get(name) {
            None => {
                namespaces.insert(name.to_string(), Some(namespace.to_string()));
            }
            Some(Some(existing)) if existing != namespace => {
                namespaces.insert(name.to_string(), None);
            }
            Some(_) => {}
        }
    }
}

// Adds a Codex-compatible namespace field to every outbound Responses function
// call whose name originated from an unambiguously flattened MCP namespace.
// This visits buffered Responses bodies and every Responses streaming event.
fn restore_responses_tool_namespaces(body: &mut Value, namespaces: &HashMap<String, String>) {
    if namespaces.is_empty() {
        return;
    }
    match body {
        Value::Array(values) => {
            for value in values {
                restore_responses_tool_namespaces(value, namespaces);
            }
        }
        Value::Object(object) => {
            if object.get("type").and_then(Value::as_str) == Some("function_call")
                && let Some(name) = object.get("name").and_then(Value::as_str)
                && let Some(namespace) = namespaces.get(name)
            {
                object
                    .entry("namespace".to_string())
                    .or_insert_with(|| Value::String(namespace.clone()));
            }
            for value in object.values_mut() {
                restore_responses_tool_namespaces(value, namespaces);
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{WireFormat, responses_tool_namespaces, restore_responses_tool_namespaces};

    #[test]
    fn restores_codex_mcp_namespace_on_responses_function_call() {
        let request = json!({
            "tools": [{
                "type": "namespace",
                "name": "mcp__open_websearch__",
                "tools": [{
                    "type": "function",
                    "name": "search",
                    "parameters": {"type": "object"}
                }]
            }]
        });
        let namespaces = responses_tool_namespaces(&request, WireFormat::OpenAiResponses);
        let mut response = json!({
            "output": [{
                "type": "function_call",
                "name": "search",
                "arguments": "{\"q\":\"Rust\"}"
            }]
        });

        restore_responses_tool_namespaces(&mut response, &namespaces);

        assert_eq!(
            response["output"][0]["namespace"],
            "mcp__open_websearch__"
        );
    }

    #[test]
    fn skips_ambiguous_codex_mcp_tool_names() {
        let request = json!({
            "tools": [
                {
                    "type": "namespace",
                    "name": "mcp__first__",
                    "tools": [{"type": "function", "name": "search"}]
                },
                {
                    "type": "namespace",
                    "name": "mcp__second__",
                    "tools": [{"type": "function", "name": "search"}]
                }
            ]
        });

        assert!(responses_tool_namespaces(&request, WireFormat::OpenAiResponses).is_empty());
    }
}
