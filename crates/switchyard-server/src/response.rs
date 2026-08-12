// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Response encoding glue for libsy server endpoints.

use std::collections::HashMap;
use std::error::Error;

use axum::Json;
use axum::response::{IntoResponse, Response as HttpResponse};
use futures_util::StreamExt;
use switchyard_protocol::{LlmResponse, Response as AlgorithmResponse};
use switchyard_translation::{
    WireFormat, encode_aggregated_response, encode_stream, restore_responses_tool_namespaces,
};

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
