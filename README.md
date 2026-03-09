Multimodal Product Price Prediction

This project builds a multimodal deep learning model to predict product prices using both text descriptions and product images.

The system combines:

* **Text embeddings** from product descriptions
* **Image features** from product images
* A **fusion neural network** that learns from both modalities

The goal is to improve price prediction accuracy by leveraging multiple types of product information.

# Project Structure

```
project/
│
├── train.csv
├── test.csv
│
├── images/
│   ├── train/
│   └── test/
│
├── processed_images/      # optional faster loading
│
├── train_text_embeddings.pt
├── test_text_embeddings.pt
│
├── model.py
├── train.py
├── predict.py
│
└── README.md
```

Dataset

The dataset consists of:

### Train Dataset

| Column          | Description              |
| --------------- | ------------------------ |
| sample_id       | Unique product ID        |
| catalog_content | Product description text |
| price           | Target variable          |
| image           | Product image            |

### Test Dataset

| Column          | Description         |
| --------------- | ------------------- |
| sample_id       | Unique product ID   |
| catalog_content | Product description |
| image           | Product image       |

Images are stored separately:

```
images/train/
images/test/
```

Model Architecture

The system uses a **fusion model** combining text and image features.

Text Encoder

Text descriptions are converted into embeddings using:

`sentence-transformers/all-MiniLM-L6-v2`

Output dimension:

```
384
```

Image Encoder

Images are processed using a pretrained CNN:

```
ResNet18
```

Output dimension after projection:

```
128
```

Fusion Network

The features are concatenated and passed through fully connected layers.

```
Text Features (384)
        +
Image Features (128)
        ↓
   Concatenation
        ↓
 Fully Connected Layers
        ↓
 Price Prediction
```

Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/product-price-prediction.git
cd product-price-prediction
```

Install dependencies:

```bash
pip install torch torchvision transformers pandas tqdm pillow
```

Generating Text Embeddings

To speed up training, text embeddings are precomputed.

```python
python generate_embeddings.py
```

This will generate:

```
train_text_embeddings.pt
test_text_embeddings.pt
```

Precomputing embeddings avoids recomputing them during every training run.

# Training the Model

Run:

```python
python train.py
```

Training includes:

* Loading embeddings
* Loading images
* Training the fusion network
* Saving the trained model

# Making Predictions

Run:

```python
python predict.py
```

This will generate:

```
test_predictions.csv
```

# Performance Optimizations

Several optimizations were used to improve speed:

### Precomputed Text Embeddings

Embeddings are generated once and stored as `.pt` files.

### Image Preprocessing

Images can optionally be converted to tensors:

```
processed_images/train
processed_images/test
```

This avoids slow disk image loading during training.

### Batch Processing

Large batch sizes improve GPU utilization.

# Technologies Used

* **PyTorch**
* **Torchvision**
* **Transformers (HuggingFace)**
* **Pandas**
* **Sentence Transformers**

# Future Improvements

Possible improvements:

* Vision Transformer (ViT) instead of ResNet
* Better fusion architectures
* Cross-modal attention
* Data augmentation
* Hyperparameter tuning
* Distributed training

# Example Output

```
sample_id,predicted_price
10001,599.23
10002,349.18
10003,1129.54
```

# Author

Jiya Jain
