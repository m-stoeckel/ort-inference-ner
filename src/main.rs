use std::cmp::Ordering;
use std::path::PathBuf;

use anyhow::anyhow;
use ndarray::{Array2, Axis};
use ndarray_stats::QuantileExt;
use onnxruntime::{
    environment::Environment, tensor::OrtOwnedTensor, GraphOptimizationLevel, LoggingLevel,
};
use rust_tokenizers::tokenizer::{BertTokenizer, Tokenizer, TruncationStrategy};
use rust_tokenizers::vocab::{BertVocab, Vocab};
use rust_tokenizers::Mask;

fn values_to_array2(values: Vec<Vec<i64>>) -> anyhow::Result<ndarray::Array2<i64>> {
    let nrows = values.len();
    let ncols = values.iter().map(|t| t.len()).max().unwrap_or(0);
    let data = values
        .into_iter()
        .map(|t| zero_pad_vec_to_length(t, ncols))
        .collect::<Result<Vec<Vec<_>>, _>>()?;
    let data = data.into_iter().flatten().collect();
    let arr: Array2<i64> = Array2::from_shape_vec((nrows, ncols), data)?;
    anyhow::Ok(arr)
}

const ZERO_512: [i64; 512] = [0_i64; 512];

fn zero_pad_vec_to_length(vec: Vec<i64>, ncols: usize) -> Result<Vec<i64>, anyhow::Error> {
    let l = vec.len();
    match l.partial_cmp(&ncols) {
        Some(Ordering::Less) => Ok([&vec, &ZERO_512[..(ncols - l)]].concat()),
        Some(Ordering::Equal) => Ok(vec),
        Some(Ordering::Greater) => Err(anyhow!(format!("Vector to long to pad: {l} > {ncols}!"))),
        None => panic!("Invalid partial_cmp while padding vector!"),
    }
}

#[derive(serde::Serialize, Debug)]
enum Label {
    O,
    #[serde(rename = "B-MISC")]
    BMisc,
    #[serde(rename = "I-MISC")]
    IMisc,
    #[serde(rename = "B-PER")]
    BPer,
    #[serde(rename = "I-PER")]
    IPer,
    #[serde(rename = "B-ORG")]
    BOrg,
    #[serde(rename = "I-ORG")]
    IOrg,
    #[serde(rename = "B-LOC")]
    BLoc,
    #[serde(rename = "I-LOC")]
    ILoc,
}

impl From<usize> for Label {
    fn from(value: usize) -> Self {
        match value {
            0 => Label::O,
            1 => Label::BMisc,
            2 => Label::IMisc,
            3 => Label::BPer,
            4 => Label::IPer,
            5 => Label::BOrg,
            6 => Label::IOrg,
            7 => Label::BLoc,
            8 => Label::ILoc,
            9 => Label::ILoc,
            _ => panic!("Label index out of bounds: {value}"),
        }
    }
}

#[derive(serde::Serialize, Debug)]
struct Annotation {
    label: Label,
    begin: u32,
    end: u32,
}

impl Annotation {
    fn new(label_id: usize, begin: u32, end: u32) -> Self {
        Annotation {
            label: Label::from(label_id),
            begin,
            end,
        }
    }
}

fn main() {
    let environment = Environment::builder()
        .with_name("test")
        .with_log_level(LoggingLevel::Verbose)
        .build()
        .unwrap();
    let mut session = environment
    .new_session_builder().unwrap()
    .with_optimization_level(GraphOptimizationLevel::Basic).unwrap()
    .with_number_threads(1).unwrap()
    .with_model_from_file("/run/media/mastoeck/hot_storage/ONNX/distilbert-base-multilingual-cased-ner-hrl/model.onnx").unwrap();

    let tokenizer = BertTokenizer::from_existing_vocab(
        BertVocab::from_file(PathBuf::from("/run/media/mastoeck/hot_storage/ONNX/distilbert-base-multilingual-cased-ner-hrl/vocab.txt")).unwrap(),
        false,
        false,
    );
    let tokens = tokenizer.encode_list(
        &[
            "My name is Clara and I live in Berkeley, California.",
            "I saw Barack Obama at the White House today.",
            "Ich habe gestern die Goethe Universität in Frankfurt am Main besucht.",
            "Nader Jokhadar had given Syria the lead with a well-struck header in the seventh minute."
        ],
        512,
        &TruncationStrategy::LongestFirst,
        0,
    );

    let (input_ids, attention_mask) = get_input_ids_and_attention_masks(&tokens);
    let input_ids = values_to_array2(input_ids).unwrap();
    let attention_mask = values_to_array2(attention_mask).unwrap();
    let input_tensor = vec![input_ids, attention_mask];

    let outputs: Vec<OrtOwnedTensor<f32, _>> = session.run(input_tensor).unwrap();
    let preds: Vec<Vec<_>> = argmax_axis(&outputs[0], 2).unwrap();

    let mut annot: Vec<Vec<Annotation>> = Vec::new();
    for (ts, ps) in std::iter::zip(tokens, preds) {
        let mut s_annot: Vec<Annotation> = Vec::new();
        for ((t, m), p) in std::iter::zip(std::iter::zip(ts.token_offsets, ts.mask), ps) {
            if p == 0 {
                continue;
            }
            match (t, m) {
                (Some(o), Mask::None | Mask::Begin) => {
                    s_annot.push(Annotation::new(p, o.begin, o.end))
                }
                (Some(o), Mask::Continuation) => s_annot.last_mut().unwrap().end = o.end,
                _ => continue,
            };
        }
        annot.push(s_annot);
    }
    let json_string = serde_json::to_string_pretty(&annot).unwrap();
    println!("{json_string}");
}

fn argmax_axis(
    outputs: &OrtOwnedTensor<'_, '_, f32, ndarray::Dim<ndarray::IxDynImpl>>,
    dim: usize,
) -> Result<Vec<Vec<usize>>, ndarray_stats::errors::MinMaxError> {
    outputs
        .map_axis(Axis(dim), |v| v.argmax())
        .rows()
        .into_iter()
        .map(|r| r.to_vec().into_iter().collect::<Result<Vec<_>, _>>())
        .collect()
}

fn get_input_ids_and_attention_masks(
    tokens: &Vec<rust_tokenizers::TokenizedInput>,
) -> (Vec<Vec<i64>>, Vec<Vec<i64>>) {
    let mut input_ids: Vec<Vec<i64>> = Vec::with_capacity(tokens.len());
    let mut attention_mask: Vec<Vec<i64>> = Vec::with_capacity(tokens.len());
    for t in tokens.iter() {
        input_ids.push(t.token_ids.clone());
        attention_mask.push(t.token_ids.iter().map(|v| (*v > 0) as i64).collect());
    }
    (input_ids, attention_mask)
}
