dependencies = ['torch', 'dp']

def phonemizer():
    import torch
    import os
    from dp.phonemizer import Phonemizer

    # Download
    hub_dir = torch.hub.get_dir()
    model_dir = os.path.join(hub_dir, 'checkpoints', 'phonemizer.pt')
    torch.hub.download_url_to_file("https://public-asai-dl-models.s3.eu-central-1.amazonaws.com/DeepPhonemizer/en_us_cmudict_ipa_forward.pt", model_dir)

    # Load
    model = Phonemizer.from_checkpoint(model_dir)
    return model
            