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
        Some(Ordering::Greater) => Err(anyhow!("...")),
        None => Err(anyhow!("...")),
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
    // println!("{tokens:?}");

    // let (input_ids, type_ids) = get_input_and_type_ids(tokens);
    // let type_ids = values_to_array2(type_ids).unwrap();
    let (input_ids, attention_mask) = get_input_ids_and_attention_masks(&tokens);
    let input_ids = values_to_array2(input_ids).unwrap();
    let attention_mask = values_to_array2(attention_mask).unwrap();
    let input_tensor = vec![input_ids, attention_mask];
    // println!("{input_tensor:?}");

    let outputs: Vec<OrtOwnedTensor<f32, _>> = session.run(input_tensor).unwrap();
    let preds: Vec<Vec<_>> = argmax_axis_2(&outputs[0]).unwrap();
    println!("{:?}", preds);
    // let softmax = outputs[0].softmax(Axis(2));
    // println!("{softmax:?}");

    let label_map: Vec<String> = vec![
        "O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC",
    ]
    .into_iter()
    .map(String::from)
    .collect();

    let mut annot: Vec<Vec<(String, String)>> = Vec::new();
    for (ts, ps) in std::iter::zip(tokens, preds) {
        let mut s_annot: Vec<(String, String)> = Vec::new();
        for ((t, m), p) in std::iter::zip(
            std::iter::zip(tokenizer.decode_to_vec(&ts.token_ids, false), ts.mask),
            ps,
        ) {
            match m {
                rust_tokenizers::Mask::None => s_annot.push((t, label_map[p].clone())),
                rust_tokenizers::Mask::Begin => s_annot.push((t, label_map[p].clone())),
                rust_tokenizers::Mask::Continuation => s_annot.push((t, label_map[p].clone())),
                _ => continue,
            };
        }
        annot.push(s_annot);
    }

    for ele in annot {
        println!("{ele:?}");
    }
}

fn argmax_axis_2(
    outputs: &OrtOwnedTensor<'_, '_, f32, ndarray::Dim<ndarray::IxDynImpl>>,
) -> Result<Vec<Vec<usize>>, ndarray_stats::errors::MinMaxError> {
    outputs
        .map_axis(Axis(2), |v| v.argmax())
        .rows()
        .into_iter()
        .map(|r| r.to_vec().into_iter().collect::<Result<Vec<_>, _>>())
        .collect()
}

fn get_input_and_type_ids(
    tokens: &Vec<rust_tokenizers::TokenizedInput>,
) -> (Vec<Vec<i64>>, Vec<Vec<i64>>) {
    let mut input_ids: Vec<Vec<i64>> = Vec::with_capacity(tokens.len());
    let mut type_ids: Vec<Vec<i64>> = Vec::with_capacity(tokens.len());
    for t in tokens.iter() {
        input_ids.push(t.token_ids.clone());
        type_ids.push(
            t.segment_ids
                .clone()
                .into_iter()
                .map(|x| x as i64)
                .collect(),
        );
    }
    (input_ids, type_ids)
}
fn get_input_ids_and_attention_masks(
    tokens: &Vec<rust_tokenizers::TokenizedInput>,
) -> (Vec<Vec<i64>>, Vec<Vec<i64>>) {
    let mut input_ids: Vec<Vec<i64>> = Vec::with_capacity(tokens.len());
    let mut attention_mask: Vec<Vec<i64>> = Vec::with_capacity(tokens.len());
    for t in tokens.into_iter() {
        input_ids.push(t.token_ids.clone());
        attention_mask.push(
            t.token_ids
                .clone()
                .iter()
                .map(|v| (v > &0_i64) as i64)
                .collect(),
        );
    }
    (input_ids, attention_mask)
}
