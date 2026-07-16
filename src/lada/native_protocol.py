"""Native LADA training protocol shared by CLIP and SigLIP2 ports."""

# Public LADA X-TAIL 16-shot schedule.  The backbone port may change encoder
# APIs but must not silently replace these method-level training choices.
LADA_16SHOT_EPOCHS = {
    "aircraft": 40,
    "caltech101": 10,
    "dtd": 30,
    "eurosat": 100,
    "flowers": 30,
    "food101": 5,
    "mnist": 200,
    "oxford_pets": 10,
    "stanford_cars": 30,
    "sun397": 10,
}

LADA_NATIVE_DEFAULTS = {
    "batch_size": 64,
    "lr": 1e-3,
    "weight_decay": 5e-4,
    "lada_k": 16,
    "prototype_k": 4,
    "image_prototypes_weight_coef": 64.0,
    "text_adapter_dim": 16,
}
