# Counterfactual Explanation with StyleGAN2

Lightweight notes on what is inside the src folder and how to run it.

## Model snapshot
- Attribute classifier built on ResNet18 with CBAM attention for multi-attribute CelebA labels (see src/classifiers).
- StyleGAN2-based synthesis with a HR-VQVAE as baseline latent space stack for counterfactual edits and latent manipulation (see src/synthesis and src/unsupervised_latentspace).
- Explanation utilities: Integrated Gradients for per-attribute saliency (see src/classifiers/integrated_gradients.py, src/classifiers/visualizations.py).

## Quick start
1) **Train first in the notebook**: open running_script.ipynb, set your data paths, and run the cells to train and save checkpoints before doing any inference.
2) **Point to your data and weights**: update CSV_PATH, IMAGE_PATH, and PATH_CHECKPOINT in src/classifiers/example_usage.py (and any scripts you run) to match your local paths and trained weights.
3) **Run the inference demo**: after training, execute:
	```bash
	python src/classifiers/example_usage.py
	```
	This loads the best checkpoint, runs a sample image through the classifier, and calls Integrated Gradients for one attribute.
4) **Use the inference utility directly**: src/classifiers/inference.py exposes inference_single_image to get probabilities and binary predictions for your own image and model.

## Sample inference outputs
- Console log lists each attribute with ground truth (if available), predicted probability, binary decision, and per-image accuracy.
- Visual explanations (Integrated Gradients heatmaps and overlays) pop up via matplotlib; add plt.savefig(...) in the visualization helpers if you want to persist the figures.
