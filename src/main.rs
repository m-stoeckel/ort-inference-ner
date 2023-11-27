use std::cmp::Ordering;
use std::path::PathBuf;

use anyhow::anyhow;
use clap::{Parser, ValueEnum};
use ndarray::{Array2, Axis};
use ndarray_stats::QuantileExt;
use ort::{Session, CUDAExecutionProvider, ExecutionProvider, TensorRTExecutionProvider};
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

#[derive(Debug, PartialEq, Eq)]
enum Entity {
    MISC,
    PER,
    ORG,
    LOC
}

#[derive(Debug)]
enum Label {
    O,
    B(Entity),
    I(Entity)
}

impl From<usize> for Label {
    fn from(value: usize) -> Self {
        match value {
            0 => Label::O,
            1 => Label::B(Entity::MISC),
            2 => Label::I(Entity::MISC),
            3 => Label::B(Entity::PER),
            4 => Label::I(Entity::PER),
            5 => Label::B(Entity::ORG),
            6 => Label::I(Entity::ORG),
            7 => Label::B(Entity::LOC),
            8 => Label::I(Entity::LOC),
            _ => panic!("Label index out of bounds: {value}"),
        }
    }
}

impl serde::Serialize for Label {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer {
        match self {
            Label::O => serializer.serialize_str("O"),
            // Label::B(e) => serializer.serialize_str(&format!("B-{e:?}")),
            // Label::I(e) => serializer.serialize_str(&format!("I-{e:?}")),
            Label::B(e) => serializer.serialize_str(&format!("{e:?}")),
            Label::I(e) => serializer.serialize_str(&format!("{e:?}")),
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
    fn new(label: Label, begin: u32, end: u32) -> Self {
        Annotation {
            label,
            begin,
            end,
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Debug)]
enum ImplementedProviders{
    CPU,
    CUDA,
    TensorRT,
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long, default_value="cpu")]
    provider: ImplementedProviders,

    #[arg(short, long, default_value_t = 0)]
    device: u8,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    
    let builder = Session::builder()?;

    match args.provider {
        ImplementedProviders::CPU => (),
        ImplementedProviders::CUDA => {
            let cuda = CUDAExecutionProvider::default();
            if let Err(err) = cuda.register(&builder) {
                Err(anyhow!("Failed to register CUDA execution provider: {err:?}"))?
            }
        }
        ImplementedProviders::TensorRT => {
            let tensor = TensorRTExecutionProvider::default();
            if let Err(err) = tensor.register(&builder) {
                Err(anyhow!("Failed to register TensorRT execution provider: {err:?}"))?
            }
        },

    }

    ort::init()
        // .with_execution_providers([CPUExecutionProvider::default().build()])
        .commit()?;

    let session = builder
		// .with_optimization_level(GraphOptimizationLevel::Level1)?
        .with_parallel_execution(true)?
		.with_intra_threads(12)?
        .with_model_from_file("/run/media/mastoeck/hot_storage/ONNX/distilbert-base-multilingual-cased-ner-hrl/model.onnx")?;

    let tokenizer = BertTokenizer::from_existing_vocab(
        BertVocab::from_file(PathBuf::from("/run/media/mastoeck/hot_storage/ONNX/distilbert-base-multilingual-cased-ner-hrl/vocab.txt"))?,
        false,
        false,
    );
    let corpus: Vec<String> =
        std::fs::read_to_string("/hot_storage/Data/Leipzig/deu/deu_wikipedia_2021_10K-1k_shuf.txt")
            .unwrap()
            .lines()
            .map(String::from)
            .filter(|s| !s.is_empty())
            .collect();
    // let corpus = [
    //     "My name is Clara and I live in Berkeley, California.",
    //     "I saw Barack Obama at the White House today.",
    //     "Ich habe gestern die Goethe Universität in Frankfurt am Main besucht.",
    //     "Nader Jokhadar had given Syria the lead with a well-struck header in the seventh minute.",
    // ];

    let annot: Vec<Vec<Annotation>> = corpus
        .chunks(128)
        .map(|batch| {
            let tokens = tokenizer.encode_list(
                batch,
                512,
                &TruncationStrategy::LongestFirst,
                0,
            );

            let (input_ids, attention_mask) = get_input_ids_and_attention_masks(&tokens);
            let input_ids:Array2<i64> = values_to_array2(input_ids).unwrap();
            let attention_mask:Array2<i64> = values_to_array2(attention_mask).unwrap();

            (input_ids, attention_mask, tokens)
        })
        // .into_iter()
        // .map(|batch| {
        //     let tokens = tokenizer.encode(
        //         &batch,
        //         None,
        //         512,
        //         &TruncationStrategy::LongestFirst,
        //         0,
        //     );
        //     let tokens = vec![tokens];

        //     let (input_ids, attention_mask) = get_input_ids_and_attention_masks(&tokens);
        //     let input_ids:Array2<i64> = values_to_array2(input_ids).unwrap();
        //     let attention_mask:Array2<i64> = values_to_array2(attention_mask).unwrap();

        //     (input_ids, attention_mask, tokens)
        // })
        .map(|(input_ids, attention_mask, tokens)|{
            let outputs  = session.run(ort::inputs![input_ids, attention_mask]?)?;

            let preds: Vec<Vec<usize>> = outputs[0]
                .extract_tensor::<f32>()?
                .view()
                .map_axis(Axis(2), |v| v.argmax())
                .rows()
                .into_iter()
                .map(|r| {
                    r.to_vec().into_iter().collect::<Result<Vec<usize>, _>>()
                })
                .collect::<Result<Vec<Vec<usize>>, _>>()?;
            
            anyhow::Ok((preds, tokens))
        })
        .filter_map(anyhow::Result::ok)
        .flat_map(|(preds, tokens)| std::iter::zip(tokens, preds))
        .map(|(ts, ps)|{
            let mut annotations: Vec<Annotation> = Vec::new();
            for ((t, m), p) in std::iter::zip(std::iter::zip(&ts.token_offsets, &ts.mask), &ps) {
                let label = Label::from(*p);
                match (t, m) {
                    (_, Mask::Special) => continue,
                    (Some(o), Mask::Continuation) => match annotations.last_mut() {
                        Some(last_annotation) => last_annotation.end = o.end,
                        None => panic!("Got a continuation token without preceeding ordinary token! {ts:#?} {ps:#?}"),
                    },
                    (Some(o), _) => {
                        if let Some(last_annotation) = annotations.last_mut() {
                            match (&last_annotation.label, &label) {
                                (Label::B(e) | Label::I(e), Label::I(n)) if e == n => {
                                    last_annotation.end = o.end;
                                    continue;
                                }
                                _ => ()
                            }
                        }
                        annotations.push(Annotation::new(label, o.begin, o.end))
                    }
                    _ => continue,
                };
            }
            
            annotations
        }).map(|annotations| {
            annotations.into_iter()
                .filter(|a|!matches!(a.label, Label::O))
                .collect::<Vec<Annotation>>()
        })
        .filter(|annotations| !annotations.is_empty())
        .collect();
    let json_string = serde_json::to_string_pretty(&annot).unwrap();
    println!("{json_string}");
    Ok(())
}

fn get_input_ids_and_attention_masks(
    batch_encoding: &Vec<rust_tokenizers::TokenizedInput>,
) -> (Vec<Vec<i64>>, Vec<Vec<i64>>) {
    let mut input_ids: Vec<Vec<i64>> = Vec::with_capacity(batch_encoding.len());
    let mut attention_mask: Vec<Vec<i64>> = Vec::with_capacity(batch_encoding.len());
    for seq_encoding in batch_encoding.iter() {
        input_ids.push(seq_encoding.token_ids.clone());
        attention_mask.push(vec![1_i64; seq_encoding.token_ids.len()]);
    }
    (input_ids, attention_mask)
}
