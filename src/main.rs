use rand::Rng;
use std::convert::TryInto;
use std::fs::File;
use std::io::{self, Read};
use std::time::Instant;
use std::usize;

const INPUT_SIZE: usize = 784; // Size of the input (28x28 pixels for MNIST)
const HIDDEN_SIZE: usize = 128; // Number of neurons in the hidden layer
const OUTPUT_SIZE: usize = 10; // Number of output classes (10 digits)
const LEARNING_RATE: f64 = 0.0005;
const EPOCHS: usize = 20;
const TRAIN_SPLIT: f64 = 0.8; // Fraction of data to use for training

const TRAIN_IMG_PATH: &str = "./data/train-images.idx3-ubyte";
const TRAIN_LBL_PATH: &str = "./data/train-labels.idx1-ubyte";

fn read_mnist_images(filename: &str) -> io::Result<(Vec<u8>, usize)> {
    const IMAGE_SIZE: usize = 28; // Assuming MNIST images are 28x28

    let mut buffer = Vec::new();
    File::open(filename)?.read_to_end(&mut buffer)?;

    // Ensure the file is large enough
    if buffer.len() < 16 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "File too small"));
    }

    // Parse header using big-endian integers
    let _magic_number = u32::from_be_bytes(buffer[0..4].try_into().unwrap());
    let n_images = u32::from_be_bytes(buffer[4..8].try_into().unwrap());
    let rows = u32::from_be_bytes(buffer[8..12].try_into().unwrap());
    let cols = u32::from_be_bytes(buffer[12..16].try_into().unwrap());

    // Validate dimensions
    assert_eq!(rows as usize, IMAGE_SIZE, "Unexpected number of rows");
    assert_eq!(cols as usize, IMAGE_SIZE, "Unexpected number of cols");

    // Extract image data
    let image_data = buffer[16..].to_vec();

    Ok((image_data, n_images.try_into().unwrap()))
}

#[derive(Debug)]
struct Layer {
    input_size: usize,
    output_size: usize,
    weights: Vec<f64>,
    biases: Vec<f64>,
}

#[derive(Debug)]
struct Network {
    hidden: Layer,
    output: Layer,
}

impl Layer {
    fn new(input_size: usize, output_size: usize) -> Self {
        let n = input_size * output_size;
        let scale = (2.0 / input_size as f64).sqrt();
        let mut rng = rand::thread_rng();
        // Initialize weights with scaled random values
        let weights: Vec<f64> = (0..n)
            .map(|_| (rng.gen::<f64>() - 0.5) * 2.0 * scale)
            .collect();

        // Initialize biases with zeros
        let biases = vec![0.0; output_size];

        Layer {
            input_size,
            output_size,
            weights,
            biases,
        }
    }
    fn forward(&self, input: &[f64], output: &mut [f64]) {
        for i in 0..self.output_size {
            output[i] = self.biases[i];
        }
        for j in 0..self.input_size {
            let in_j = input[j];
            let weight_row = &self.weights[j * self.output_size..(j + 1) * self.output_size];
            for i in 0..self.output_size {
                output[i] += in_j * weight_row[i];
            }
        }

        // Step 2: Apply ReLU activation function to the output
        output.iter_mut().for_each(|x| *x = x.max(0.0)); // ReLU activation
    }
}

impl Network {
    fn new() -> Self {
        let hidden = Layer::new(INPUT_SIZE, HIDDEN_SIZE);
        let output = Layer::new(HIDDEN_SIZE, OUTPUT_SIZE);

        Network { hidden, output }
    }
}

fn softmax(output: &mut [f64]) {
    let max = output.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut sum = 0.0;

    for i in 0..output.len() {
        output[i] = (output[i] - max).exp();
        sum += output[i];
    }

    for i in 0..output.len() {
        output[i] /= sum;
    }
}

fn backward(
    layer: &mut Layer,
    input: &[f64],
    output_grad: &[f64],
    mut input_grad: Option<&mut [f64]>,
    lr: f64,
) {
    for i in 0..layer.output_size {
        for j in 0..layer.input_size {
            let idx = j * layer.output_size + i;
            let grad = output_grad[i] * input[j];

            // Update weights
            layer.weights[idx] -= lr * grad;

            // Update input gradient if provided
            if let Some(ref mut input_grad) = input_grad {
                input_grad[j] += output_grad[i] * layer.weights[idx];
            }
        }

        // Update biases
        layer.biases[i] -= lr * output_grad[i];
    }
}

fn train(net: &mut Network, input: &[f64], label: usize, lr: f64) -> Vec<f64> {
    let mut hidden_output = [0.0; HIDDEN_SIZE];
    let mut final_output = [0.0; OUTPUT_SIZE];
    let mut output_grad = [0.0; OUTPUT_SIZE];
    let mut hidden_grad = [0.0; HIDDEN_SIZE];

    // Forward Pass: Input to Hidden Layer
    net.hidden.forward(input, &mut hidden_output);
    hidden_output
        .iter_mut()
        .for_each(|x| *x = f64::max(*x, 0.0)); // ReLU Activation

    // Forward Pass: Hidden to Output Layer
    net.output.forward(&hidden_output, &mut final_output);
    let mut final_output_f64: Vec<f64> = final_output.iter().map(|&x| x as f64).collect();

    // Compute Output Gradient (Cross-Entropy Loss)
    softmax(&mut final_output_f64);

    // Compute Output Gradient (Cross-Entropy Loss)
    output_grad.iter_mut().enumerate().for_each(|(i, x)| {
        *x = final_output[i] - if i == label { 1.0 } else { 0.0 };
    });

    // Backward Pass: Output Layer to Hidden Layer
    backward(
        &mut net.output,
        &hidden_output,
        &output_grad,
        Some(&mut hidden_grad),
        lr,
    );

    // Backpropagate Through ReLU Activation (Derivatives)
    hidden_grad
        .iter_mut()
        .zip(hidden_output.iter())
        .for_each(|(grad, &output)| {
            *grad *= if output > 0.0 { 1.0 } else { 0.0 }; // ReLU Derivative
        });

    // Backward Pass: Hidden Layer to Input Layer
    backward(&mut net.hidden, &input, &hidden_grad, None, lr);

    final_output_f64
}

struct InputData {
    n_images: usize,
    images: Vec<u8>,
    labels: Vec<u8>,
}

fn read_mnist_labels(filename: &str) -> Result<Vec<u8>, io::Error> {
    let mut file = File::open(filename)?;

    // Skip the first 4 bytes (Magic number)
    let mut magic_number = [0u8; 4];
    file.read_exact(&mut magic_number)?;

    // Read the number of labels
    let mut n_labels = [0u8; 4];
    file.read_exact(&mut n_labels)?;

    // Convert from big-endian to little-endian (we need to interpret this number)
    let n_labels = u32::from_be_bytes(n_labels) as usize;

    // Allocate space for the labels
    let mut labels = vec![0u8; n_labels];

    // Read the labels
    file.read_exact(&mut labels)?;

    Ok(labels)
}

fn predict(net: &Network, input: &[f64]) -> usize {
    let mut hidden_output = [0.0; HIDDEN_SIZE];
    let mut final_output = [0.0; OUTPUT_SIZE];

    // Forward Pass: Input to Hidden Layer
    net.hidden.forward(input, &mut hidden_output);

    // Forward Pass: Hidden to Output Layer
    net.output.forward(&hidden_output, &mut final_output);

    // Apply Softmax to final_output
    softmax(&mut final_output);

    // Find the index of the maximum value in final_output
    let mut max_index = 0;
    for i in 1..OUTPUT_SIZE {
        if final_output[i] > final_output[max_index] {
            max_index = i;
        }
    }

    max_index
}

fn main() {
    let mut net = Network::new();
    let mut data = InputData {
        images: vec![],
        labels: vec![],
        n_images: 0,
    };

    let learning_rate = LEARNING_RATE;
    let mut img = vec![0.0; INPUT_SIZE];

    // Read and shuffle data
    println!("Reading MNIST data");
    (data.images, data.n_images) = read_mnist_images(TRAIN_IMG_PATH).unwrap();
    println!("Finished reading {} images.", data.n_images);
    println!("Finished reading mnist images: {}", data.n_images);
    data.labels = read_mnist_labels(TRAIN_LBL_PATH).unwrap();
    println!("Finished reading {} labels.", data.labels.len());
    // shuffle_data(&mut data.images, &mut data.labels, data.n_images);

    let train_size = (data.n_images as f64 * TRAIN_SPLIT) as usize;
    let test_size = data.n_images - train_size;

    for epoch in 0..EPOCHS {
        let start = Instant::now();
        let mut total_loss = 0.0;
        println!("Epoch {}: Training started...", epoch + 1);

        for i in 0..train_size {
            for k in 0..INPUT_SIZE {
                img[k] = data.images[i * INPUT_SIZE + k] as f64 / 255.0; // Normalize image
            }

            let final_output = train(
                &mut net,
                &img,
                (*data.labels.get(i).unwrap()).into(),
                learning_rate,
            );
            let index: usize = (*data.labels.get(i).unwrap()).into();
            total_loss += -(final_output[index] + 1e-10); // Cross-Entropy Loss
        }

        let correct = (train_size..data.n_images)
            .filter(|&i| {
                let mut img = vec![0.0; INPUT_SIZE];
                for k in 0..INPUT_SIZE {
                    img[k] = data.images[i * INPUT_SIZE + k] as f64 / 255.0;
                }
                predict(&net, &img) == data.labels[i].into()
            })
            .count();

        let duration = start.elapsed();
        let accuracy = correct as f64 / test_size as f64 * 100.0;
        let avg_loss = total_loss / train_size as f64;

        println!(
            "Epoch {}, Accuracy: {:.2}%, Avg Loss: {:.4}, Time: {:.2?}",
            epoch + 1,
            accuracy,
            avg_loss,
            duration
        );
    }
}
