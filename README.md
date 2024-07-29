# Inference Speed Enhancement for Open-Source LLMs

## Project Overview

This project aims to significantly boost the inference speed of open-source large language models (LLMs) by transitioning from the traditional Multi-head Attention mechanism to the Grouped Query Attention mechanism. The approach is based on insights from the paper *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*.

**Note:** This is an ongoing project, and development is actively underway.

## Key Objectives

- **Transition Mechanism**: Move from Multi-head Attention to Grouped Query Attention.
- **Evaluation**: Assess model performance using the SQuAD v2 dataset.
- **Refinement**: Apply mean pooling to key and value components and utilize Low-Rank Adaptation for fine-tuning.
- **Performance Measurement**: Evaluate both the quality of generated responses and inference speed.

## Workflow

1. **Evaluation**
   - Evaluate the model's responses to the SQuAD v2 dataset.
   - Use cosine similarity to compare responses with expected answers.
   - Perform preliminary checks with OpenAI's ADA model.
   - For scores in the range of 0.70 to 0.85, conduct additional verification using GPT-3.5.

2. **Refinement**
   - Mean pool weights of key and value components in the Multi-head Attention mechanism using a custom script.
   - Utilize Low-Rank Adaptation for fine-tuning:
     - **Unsupervised Fine-Tuning**: On the Slim Pajama dataset.
     - **Instruction-Based Fine-Tuning**: On the Awesome dataset.

3. **Performance Measurement**
   - Measure and analyze the quality of generated responses.
   - Assess the speed of inference.

## Repository Structure

- **`evaluation/`**: Contains files for model evaluation.
- **`training/`**: Contains files for model training.
