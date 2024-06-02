import logging
import os
from pprint import pprint
import string

import hydra
from hydra.core.config_store import ConfigStore
from zipp import Path

from deepspeech_pytorch.configs.inference_config import ServerConfig

# DONT touch - BASE_DIR not visible if this code runs after the below imports
cs = ConfigStore.instance()
cs.store(name="config", node=ServerConfig)

from pathlib import Path

import Levenshtein
import requests
from hydra.utils import get_original_cwd


def compute_WER(output: str, reference: str):
    """
    Computes the Word Erro Rate, defined as the edit distance between the two provided
    sentences after tokenizing to words.

    Args:
        output (str): Prediction string separated by spaces
        reference (str): Ground truth
    """
    # build mapping of words to integers
    b = set(output.split() + reference.split())
    word2char = dict(zip(b, range(len(b))))

    # map the words to a char array (Levenhstein packages only accepts strings)
    w1 = [chr(word2char[w])for w in output.split()]
    w2 = [chr(word2char[w])for w in reference.split()]

    return Levenshtein.distance(''.join(w1), ''.join(w2))

def compute_CER(output, reference):
    """
    Computes the character error rate, defined as the edit distance.

    Args:
        output (str): Prediction string separated by spaces
        reference (str): Ground truth
    """
    output = output.replace(' ', '')
    reference = reference.replace(' ', '')
    return  Levenshtein.distance(output, reference)

@hydra.main(config_name="config")
def main(cfg: ServerConfig):
    os.chdir(get_original_cwd())
    logging.info(os.getcwd())
    
    API_POST_ENDPOINT = 'http://localhost:8888/transcribe'
    BASE_PATH = Path('./prediction')

    wav_files = [file.name for file in (BASE_PATH / "wavs").iterdir() if file.suffix == '.wav']

    for index, wav_file in enumerate(wav_files):
        logging.info(f"Reading file number: {index}")

        wav_path = BASE_PATH / "wavs" / wav_file
        transcript_path = BASE_PATH / "transcripts" / wav_file.replace(".wav", ".txt")

        logging.debug(f"Wav path: {wav_path}")
        logging.debug(f"Transcript path: {transcript_path}")

        if not wav_path.exists():
            raise FileNotFoundError(f"File {wav_path} does not exist.")
        if not transcript_path.exists():
            raise FileNotFoundError(f"File {transcript_path} does not exist.")

        with open(wav_path, 'rb') as wav_file, \
            open(transcript_path, 'r') as transcript_file:
            # Make the POST request
            reference = transcript_file.read()
            # Remove punctuation and make lowercase
            reference = reference.translate(str.maketrans('', '', string.punctuation)).lower()

            response = requests.post(API_POST_ENDPOINT,
                                     files={'file': (wav_file.name, wav_file)})
            # Print the response
            response = response.json()
            prediction = response["transcription"]

            wer_instance = compute_WER(prediction, reference)
            cer_instance = compute_CER(prediction, reference)
            num_tokens = len(reference.split())
            num_chars = len(reference.replace(' ', ''))
            WER = (float(wer_instance) / num_tokens) * 100
            CER = (float(cer_instance) / num_chars) * 100

            logging.info(f"Ground truth: {reference}")
            logging.info(f"Model output: {prediction}")

            logging.info(f"\t\tWER: {WER:.2f}%")
            logging.info(f"\t\tCER: {CER:.2f}%")

if __name__ == "__main__":
    main()
