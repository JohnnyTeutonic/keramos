//! Handcrafted regression from IR_SPEC 5.14; no conformance fixtures.
use crate::{json, model::Model, ops::{self, Ctx, Env, Rows, Slots, Val}, tensor::Tensor};
use std::collections::HashMap;

#[test]
fn longrope_uses_session_length_for_routed_subset() {
    let doc = json::parse_value(r#"{"model":{"vocab":2,"d":2,"ctx":8},"embed":{"nodes":[]},"block":{"repeat":1,"state":[],"nodes":[]},"head":{"nodes":[]}}"#).unwrap();
    let mut weights = HashMap::new();
    weights.insert("short".into(), Tensor { shape: vec![1], data: vec![1.0] });
    weights.insert("long".into(), Tensor { shape: vec![1], data: vec![2.0] });
    let model = Model::from_doc(&doc, weights, 0).unwrap();
    let node = json::parse_value(r#"{"op":"rope","in":["q","k","pos"],"out":["qr","kr"],"heads":1,"head_dim":2,"theta":10000,"rope_heads":"all","pairing":"interleaved","freq_factors":{"short":"short","long":"long","original_ctx":2}}"#).unwrap();
    // A later recursion step contains positions 0 and 1, while positions 2 and 3 halted.
    // Both rows still belong to a four-token session, so the long divisor is 2.
    for sequence_len in [4, 2] {
        let rows = Rows { pos: vec![0, 1], sequence_len };
        let ctx = Ctx { model: &model, app: Some(0), rows: &rows, codes: None, section: 1, node_index: 0 };
        let mut env = Env::new();
        for name in ["q", "k"] { env.insert(name.into(), Val::T(Tensor::mat(2, 2, vec![1.0, 0.0, 1.0, 0.0]))); }
        env.insert("pos".into(), Val::I(vec![0, 1]));
        ops::exec(&ctx, &node, &mut env, &mut Slots::default()).unwrap();
        let angle = if sequence_len > 2 { 0.5f32 } else { 1.0f32 };
        for name in ["qr", "kr"] {
            let result = match env.get(name).unwrap() { Val::T(t) => t, _ => panic!("expected tensor") };
            assert!((result.data[2] - angle.cos()).abs() < 1e-7);
            assert!((result.data[3] - angle.sin()).abs() < 1e-7);
        }
    }
}
