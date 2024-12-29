use rand::Rng;
use std::fs::File;
use std::io::{self, Read};
use std::time::Instant;

const INPUT_SIZE: usize = 784;
const HIDDEN_SIZE: usize = 256; 
const OUTPUT_SIZE: usize = 10;
const LEARNING_RATE: f32 = 0.0005; // Changed to f32
const MOMENTUM: f32 = 0.9;
const EPOCHS: usize = 20;
const TRAIN_SPLIT: f32 = 0.8;
const PRINT_INTERVAL: usize = 1000; // Print every 1000 samples



const TRAIN_IMG_PATH: &str = "./data/train-images.idx3-ubyte";
const TRAIN_LBL_PATH: &str = "./data/train-labels.idx1-ubyte";

#[derive(Debug)]
struct Layer {
    weights: Vec<f32>,
    biases: Vec<f32>,
    weight_momentum: Vec<f32>,
    bias_momentum: Vec<f32>,
    input_size: usize,
    output_size: usize,
}

impl Layer {
    fn new(input_size: usize, output_size: usize) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (2.0 / input_size as f32).sqrt();
        
        Layer {
            weights: (0..input_size * output_size)
                .map(|_| (rng.gen::<f32>() - 0.5) * 2.0 * scale)
                .collect(),
            biases: vec![0.0; output_size],
            weight_momentum: vec![0.0; input_size * output_size],
            bias_momentum: vec![0.0; output_size],
            input_size,
            output_size,
        }
    }

    #[inline(always)]
    fn forward(&self, input: &[f32], output: &mut [f32]) {
        output.copy_from_slice(&self.biases);

        for j in 0..self.input_size {
            let in_j = input[j];
            let weight_slice = &self.weights[j * self.output_size..(j + 1) * self.output_size];
            for i in 0..self.output_size {
                output[i] += in_j * weight_slice[i];
            }
        }

        // ReLU activation
        for val in output.iter_mut() {
            *val = val.max(0.0);
        }
    }

    #[inline(always)]
    fn backward(&mut self, input: &[f32], output_grad: &[f32], learning_rate: f32) {
        for j in 0..self.input_size {
            let in_j = input[j];
            let base_idx = j * self.output_size;
            
            for i in 0..self.output_size {
                let idx = base_idx + i;
                let grad = output_grad[i] * in_j;
                self.weight_momentum[idx] = MOMENTUM * self.weight_momentum[idx] + learning_rate * grad;
                self.weights[idx] -= self.weight_momentum[idx];
            }
        }

        for i in 0..self.output_size {
            self.bias_momentum[i] = MOMENTUM * self.bias_momentum[i] + learning_rate * output_grad[i];
            self.biases[i] -= self.bias_momentum[i];
        }
    }
}

struct Network {
    hidden: Layer,
    output: Layer,
    // Pre-allocate buffers
    hidden_output: Vec<f32>,
    final_output: Vec<f32>,
    output_grad: Vec<f32>,
    hidden_grad: Vec<f32>,
}

impl Network {
    fn new() -> Self {
        Network {
            hidden: Layer::new(INPUT_SIZE, HIDDEN_SIZE),
            output: Layer::new(HIDDEN_SIZE, OUTPUT_SIZE),
            hidden_output: vec![0.0; HIDDEN_SIZE],
            final_output: vec![0.0; OUTPUT_SIZE],
            output_grad: vec![0.0; OUTPUT_SIZE],
            hidden_grad: vec![0.0; HIDDEN_SIZE],
        }
    }

    #[inline(always)]
    fn train(&mut self, input: &[f32], label: usize, learning_rate: f32) -> f32 {
        // Forward pass
        self.hidden.forward(input, &mut self.hidden_output);
        self.output.forward(&self.hidden_output, &mut self.final_output);
        
        // Softmax and cross-entropy
        let mut max = self.final_output[0];
        for &val in &self.final_output[1..] {
            max = max.max(val);
        }

        let mut sum = 0.0;
        for val in self.final_output.iter_mut() {
            *val = (*val - max).exp();
            sum += *val;
        }

        for val in self.final_output.iter_mut() {
            *val /= sum;
        }

        // Compute gradients
        for i in 0..OUTPUT_SIZE {
            self.output_grad[i] = self.final_output[i] - (if i == label { 1.0 } else { 0.0 });
        }

        // Backward pass
        self.output.backward(&self.hidden_output, &self.output_grad, learning_rate);

        // Compute hidden gradients
        self.hidden_grad.fill(0.0);
        for i in 0..HIDDEN_SIZE {
            if self.hidden_output[i] > 0.0 {  // ReLU derivative
                let weight_slice = &self.output.weights[i * OUTPUT_SIZE..(i + 1) * OUTPUT_SIZE];
                for j in 0..OUTPUT_SIZE {
                    self.hidden_grad[i] += weight_slice[j] * self.output_grad[j];
                }
            }
        }

        self.hidden.backward(input, &self.hidden_grad, learning_rate);

        -self.final_output[label].ln()
    }

    fn predict(&mut self, input: &[f32]) -> usize {
        self.hidden.forward(input, &mut self.hidden_output);
        self.output.forward(&self.hidden_output, &mut self.final_output);
        
        // Simple argmax
        let mut max_idx = 0;
        let mut max_val = self.final_output[0];
        for (i, &val) in self.final_output.iter().enumerate().skip(1) {
            if val > max_val {
                max_val = val;
                max_idx = i;
            }
        }
        max_idx
    }
}

// Helper functions remain mostly the same
fn read_mnist_images(filename: &str) -> io::Result<(Vec<u8>, usize)> {
    let mut buffer = Vec::new();
    File::open(filename)?.read_to_end(&mut buffer)?;

    let n_images = u32::from_be_bytes(buffer[4..8].try_into().unwrap()) as usize;
    Ok((buffer[16..].to_vec(), n_images))
}

fn read_mnist_labels(filename: &str) -> io::Result<Vec<u8>> {
    let mut buffer = Vec::new();
    File::open(filename)?.read_to_end(&mut buffer)?;
    Ok(buffer[8..].to_vec())
}

fn main() {
    println!("\n=== Neural Network Training Started ===\n");
    println!("Configuration:");
    println!("Learning Rate: {}", LEARNING_RATE);
    println!("Momentum: {}", MOMENTUM);
    println!("Epochs: {}", EPOCHS);
    println!("Train Split: {:.1}%\n", TRAIN_SPLIT * 100.0);

    let mut network = Network::new();
    
    println!("\nLoading MNIST dataset...");
    let (images, n_images) = read_mnist_images(TRAIN_IMG_PATH).unwrap();
    let labels = read_mnist_labels(TRAIN_LBL_PATH).unwrap();
    println!("MNIST dataset loaded.");

    let train_size = (n_images as f32 * TRAIN_SPLIT) as usize;
    let test_size = n_images - train_size;
    println!("Dataset split: {} training samples, {} test samples\n", train_size, test_size);

    let mut img = vec![0.0; INPUT_SIZE];
    let progress_bar_width = 50;

    for epoch in 0..EPOCHS {
        println!("\nEpoch {} starting...", epoch + 1);
        let start = Instant::now();
        let mut total_loss = 0.0;
        let mut correct = 0;
        let mut last_print_time = Instant::now();

        // Training phase
        println!("Training phase:");
        for i in 0..train_size {
            // Progress bar and statistics
            if i % PRINT_INTERVAL == 0 {
                let progress = i as f32 / train_size as f32;
                let filled = (progress * progress_bar_width as f32) as usize;
                let bar: String = std::iter::repeat("=").take(filled)
                    .chain(std::iter::repeat("-").take(progress_bar_width - filled))
                    .collect();
                
                let elapsed = last_print_time.elapsed().as_secs_f32();
                let samples_per_sec = PRINT_INTERVAL as f32 / elapsed;
                last_print_time = Instant::now();

                print!("\r[{}] {:.1}% ({:.1} samples/sec)", 
                    bar, progress * 100.0, samples_per_sec);
                std::io::Write::flush(&mut std::io::stdout()).unwrap();
            }

            // Convert and normalize image data
            for j in 0..INPUT_SIZE {
                img[j] = images[i * INPUT_SIZE + j] as f32 / 255.0;
            }

            let label = labels[i] as usize;
            total_loss += network.train(&img, label, LEARNING_RATE);

            if network.predict(&img) == label {
                correct += 1;
            }
        }
        println!("\nTraining phase completed");

        // Testing phase
        println!("Testing phase:");
        let mut test_correct = 0;
        for i in train_size..n_images {
            if (i - train_size) % PRINT_INTERVAL == 0 {
                let progress = (i - train_size) as f32 / test_size as f32;
                let filled = (progress * progress_bar_width as f32) as usize;
                let bar: String = std::iter::repeat("=").take(filled)
                    .chain(std::iter::repeat("-").take(progress_bar_width - filled))
                    .collect();
                
                print!("\r[{}] {:.1}%", bar, progress * 100.0);
                std::io::Write::flush(&mut std::io::stdout()).unwrap();
            }

            for j in 0..INPUT_SIZE {
                img[j] = images[i * INPUT_SIZE + j] as f32 / 255.0;
            }
            if network.predict(&img) == labels[i] as usize {
                test_correct += 1;
            }
        }
        println!("\nTesting phase completed");

        let duration = start.elapsed();
        println!("\nEpoch {} Summary:", epoch + 1);
        println!("Training Accuracy: {:.2}%", correct as f32 / train_size as f32 * 100.0);
        println!("Test Accuracy: {:.2}%", test_correct as f32 / test_size as f32 * 100.0);
        println!("Average Loss: {:.4}", total_loss / train_size as f32);
        println!("Time: {:.2?}", duration);
        println!("Samples per second: {:.1}", train_size as f32 / duration.as_secs_f32());
        println!("----------------------------------------");
    }

    println!("\n=== Training Completed ===");
}
