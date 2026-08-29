"""
Dataset loading script for the De-identification (Deid) dataset.
This script processes medical text records and splits them into sentences with NER annotations.
"""

import re
import pandas as pd
import nltk
import os
from typing import Dict, Any, List
from tqdm import tqdm
import datasets
import ast

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

_CITATION = """\
@inproceedings{stubbs2015automated,
  title={Automated systems for the de-identification of longitudinal clinical narratives: Overview of 2014 i2b2/UTHealth shared task Track 1},
  author={Stubbs, Amber and Kotfila, Christopher and Xu, Hua},
  booktitle={Journal of biomedical informatics},
  volume={58},
  pages={S11--S19},
  year={2015},
  publisher={Elsevier}
}
"""

_DESCRIPTION = """\
The De-identification dataset contains medical text records with Named Entity Recognition (NER) annotations.
The dataset is processed to split records into individual sentences while preserving entity annotations.
Each sentence is tokenized and annotated in IOB format for training NER models.
"""

_URLS = {
    "annotations": "I2B2-2014-Relabeled-PhysionetGoldCorpus.csv",
    "corrections": "corrections.txt",
}

_HOMEPAGE = "https://www.i2b2.org/NLP/DataSets/"

_LICENSE = "Data User Agreement required"

class DeidConfig(datasets.BuilderConfig):
    """BuilderConfig for Deid dataset."""

    def __init__(self, **kwargs):
        """BuilderConfig for Deid.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(DeidConfig, self).__init__(**kwargs)

class Deid(datasets.GeneratorBasedBuilder):
    """De-identification dataset with sentence-level NER annotations."""

    BUILDER_CONFIGS = [
        DeidConfig(
            name="deid",
            version=datasets.Version("1.0.0"),
            description="De-identification dataset with sentence-level processing",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "record_id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "ner_tags": datasets.Sequence(datasets.Value("string")),
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""

        # Check if id.text file exists in the expected location
        id_text_path = "id.text"  
        if not os.path.exists(id_text_path):
            raise FileNotFoundError(
                f"Required file 'id.text' not found. Please download it from PhysioNet "
                f"and place it in the dataset root directory. This file requires a license "
                f"and cannot be distributed with the dataset."
            )
        
        # Download other files from the repository
        downloaded_files = dl_manager.download(_URLS)
        
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "text_file": id_text_path,
                    "annotations_file": downloaded_files["annotations"],
                    "corrections_file": downloaded_files["corrections"],
                },
            ),
        ]

    def _generate_examples(self, text_file, annotations_file, corrections_file):
        """Yields examples."""
        # Load and process the data
        id2annotations = self._load_annotations(annotations_file)
        records_txt = self._load_records(text_file)
        ds_annotations = self._map_annotations_to_records(id2annotations, records_txt)
        
        # Process with simple sentence splitting
        processed_data = self._process_records_simple_sentences(ds_annotations)
        
        # # Filter out sentences with no entities
        # processed_data = [item for item in processed_data if any(tag != 'O' for tag in item['ner_tags'])]
        #print(f"Processed data after removing negatives samples: {len(processed_data)}")
        # Apply corrections if file exists
        if corrections_file and os.path.exists(corrections_file):
            corrections_dict = self._load_corrections(corrections_file)
            processed_data = self._apply_corrections_to_dataset(processed_data, corrections_dict)

        # # Filter out sentences with no entities after corrections
        # processed_data = [item for item in processed_data if any(tag != 'O' for tag in item['ner_tags'])]
        
        # Yield examples
        for idx, item in enumerate(tqdm(processed_data)):
            yield idx, {
                "record_id": item["record_id"],
                "tokens": item["tokens"],
                "ner_tags": item["ner_tags"],
            }

    def _load_records(self, filename):
        """Load records from text file."""
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
        
        pattern = r'(START_OF_RECORD=.*?\|\|\|\|END_OF_RECORD)'
        records = re.findall(pattern, content, flags=re.DOTALL)
        cleaned_records = []
        
        for record in records:
            text = record
            for marker in ["START_OF_RECORD=", "||||END_OF_RECORD"]:
                text = text.replace(marker, "")
           
            match_res = re.match(r'^(\d+\|\|\|\|\d+\|\|\|\|)', text)
            record_id = match_res.group(1) if match_res else None
            text = re.sub(r'^\d+\|\|\|\|\d+\|\|\|\|', '', text)
            cleaned_records.append({"record_id": record_id, "text": text})
        
        return cleaned_records

    def _load_annotations(self, file_path):
        """Load annotations from CSV file."""
        annotations = pd.read_csv(file_path)
        grouped_annotations = annotations.groupby('record_id')
        
        id2annotations = {}
        for record_id, group in grouped_annotations:
            if record_id not in id2annotations:
                id2annotations[record_id] = []
            
            for el in group.itertuples():
                annotation = {
                    'start': el.begin,
                    'end': el.begin + el.length,
                    'type': el.type
                }
                id2annotations[record_id].append(annotation)
        return id2annotations

    def _map_annotations_to_records(self, id2annotations, records_txt):
        """Map annotations to records."""
        ds_annotations = []
        for item in records_txt:
            text = item["text"]
            ann_dict = {
                "record_id": item["record_id"],
                "text": text,
                "annotations": []
            }
            
            if item["record_id"] in id2annotations:
                annotations = id2annotations[item["record_id"]]
                
                for annotation in annotations:
                    start = annotation['start'] + 1 # first char not included 
                    end = annotation['end'] + 1 # last char included
                    ent_type = annotation['type']
                    ann_dict["annotations"].append({
                        "start": start,
                        "end": end,
                        "span": text[start:end],
                        "type": ent_type
                    })
                ds_annotations.append(ann_dict)
        return ds_annotations

    def _split_long_sentence(self, sentence: str, max_tokens: int = 150) -> List[str]:
        """
        Split a long sentence into smaller chunks based on punctuation and conjunctions.
        """
        # First check if sentence needs splitting
        if len(nltk.word_tokenize(sentence)) <= max_tokens:
            return [sentence]
        
        # Define split patterns in order of preference
        split_patterns = [
            # Strong punctuation marks
            r'\n+',  # Split on one or more newlines (high priority for clinical text)
            r';\s+',
            r',\s+(?:and|but|or|yet|so|for|nor)\s+',  # Comma + coordinating conjunction
            r',\s+(?:however|therefore|moreover|furthermore|nevertheless|meanwhile|consequently)\s+',  # Comma + conjunctive adverb
            # Weaker splits
            r',\s+(?:which|that|who|where|when)\s+',  # Comma + relative pronoun
            r',\s+(?:with|without|including|excluding|during|after|before)\s+',  # Comma + preposition
            r',\s+',  # Any comma (last resort for commas)
            # Final fallback - split on conjunctions without comma
            r'\s+(?:and|but|or)\s+',
        ]
        
        chunks = [sentence]
        
        for pattern in split_patterns:
            new_chunks = []
            split_made = False
            
            for chunk in chunks:
                if len(nltk.word_tokenize(chunk)) <= max_tokens:
                    new_chunks.append(chunk)
                    continue
                    
                # Try to split this chunk
                parts = re.split(f'({pattern})', chunk)
                if len(parts) > 1:
                    # Reconstruct meaningful segments
                    current_segment = ""
                    for i, part in enumerate(parts):
                        if re.match(pattern, part):  # This is a separator
                            current_segment += part
                        else:
                            if current_segment:
                                current_segment += part
                            else:
                                current_segment = part
                            
                            # Check if we should end this segment
                            if (i == len(parts) - 1 or  # Last part
                                len(nltk.word_tokenize(current_segment)) >= max_tokens * 0.7):  # Getting close to limit
                                new_chunks.append(current_segment.strip())
                                current_segment = ""
                                split_made = True
                    
                    # Add any remaining segment
                    if current_segment.strip():
                        new_chunks.append(current_segment.strip())
                else:
                    new_chunks.append(chunk)
            
            chunks = new_chunks
            # If we made splits and all chunks are now under the limit, we're done
            if split_made and all(len(nltk.word_tokenize(chunk)) <= max_tokens for chunk in chunks):
                break
        
        return [chunk.strip() for chunk in chunks if chunk.strip()]

    def _split_into_sentences(self, text: str, max_tokens: int = 150) -> List[str]:
        """
        Split text into sentences using NLTK sentence tokenizer, then further split long sentences.
        """
        sentences = nltk.sent_tokenize(text)
        
        # Further split any sentences that are still too long
        final_sentences = []
        for sentence in sentences:
            chunks = self._split_long_sentence(sentence, max_tokens)
            final_sentences.extend(chunks)
        
        return final_sentences

    def _map_annotations_to_sentences(self, original_text, sentences, annotations):
        """Map annotations to individual sentences."""
        sentence_annotations = []
        current_pos = 0
        
        for sentence in sentences:
            sentence_start = original_text.find(sentence, current_pos)
            if sentence_start == -1:
                sentence_start = original_text.find(sentence, current_pos)
                if sentence_start == -1:
                    sentence_annotations.append([])
                    continue
            
            sentence_end = sentence_start + len(sentence)
            
            sentence_anns = []
            for ann in annotations:
                ann_start = ann['start']
                ann_end = ann['end']
                
                if (ann_start < sentence_end and ann_end > sentence_start):
                    new_start = max(0, ann_start - sentence_start)
                    new_end = min(len(sentence), ann_end - sentence_start)
                    
                    if new_end > new_start:
                        sentence_anns.append({
                            'start': new_start,
                            'end': new_end,
                            'span': sentence[new_start:new_end],
                            'type': ann['type']
                        })
            
            sentence_annotations.append(sentence_anns)
            current_pos = sentence_end
        
        return sentence_annotations

    def _convert_to_iob_format(self, record_id, text, annotations):
        """Convert text and annotations to IOB format."""
        tokens = nltk.word_tokenize(text)
        ner_tags = ['O'] * len(tokens)
        
        # Create character to token mapping
        char_to_token = {}
        current_pos = 0
        
        for token_idx, token in enumerate(tokens):
            token_start = text.find(token, current_pos)
            if token_start != -1:
                for char_pos in range(token_start, token_start + len(token)):
                    char_to_token[char_pos] = token_idx
                current_pos = token_start + len(token)
        
        # Process annotations
        for annotation in annotations:
            start_char = annotation['start']
            end_char = annotation['end']
            label = annotation['type']
            
            overlapping_tokens = set()
            for char_pos in range(start_char, end_char):
                if char_pos in char_to_token:
                    overlapping_tokens.add(char_to_token[char_pos])
            
            overlapping_tokens = sorted(list(overlapping_tokens))
            
            for i, token_idx in enumerate(overlapping_tokens):
                if i == 0:
                    ner_tags[token_idx] = f'B-{label}'
                else:
                    ner_tags[token_idx] = f'I-{label}'
        
        return {
            'record_id': record_id,
            'tokens': tokens,
            'ner_tags': ner_tags
        }

    def _process_records_simple_sentences(self, records_with_annotations):
        """Process records by splitting into sentences."""
        processed_sentences = []
        
        for record in records_with_annotations:
            record_id = record['record_id']
            text = record['text']
            annotations = record['annotations']
            
            sentences = self._split_into_sentences(text)
            sentence_annotations = self._map_annotations_to_sentences(text, sentences, annotations)
            
            for i, (sentence, sentence_anns) in enumerate(zip(sentences, sentence_annotations)):
                sentence_record_id = f"{record_id}_sent_{i}"
                iob_data = self._convert_to_iob_format(sentence_record_id, sentence, sentence_anns)
                processed_sentences.append(iob_data)
        
        return processed_sentences

    def _load_corrections(self, corrections_file):
        """Load corrections from JSONL file."""
        import json
        with open(corrections_file) as f:
            corrections = f.readlines()
            corrections = [eval(line) for line in corrections if line.strip()]
        
        corrections_dict = {}
        for correction in corrections:
            record_id = correction['id']
            if record_id not in corrections_dict:
                corrections_dict[record_id] = []
            corrections_dict[record_id].append(correction)
        return corrections_dict

    def _apply_corrections_to_dataset(self, dataset, corrections_dict):
        """Apply corrections to the dataset."""
        corrected_dataset = []
        #print("Correction to apply:", corrections_dict)
    
        for item in dataset:
            # Create a deep copy to avoid modifying the original
            corrected_item = {
                'record_id': item['record_id'],
                'tokens': item['tokens'][:],  # Copy the list
                'ner_tags': item['ner_tags'][:]  # Copy the list
            }
            
            record_id = item['record_id']
            
            if record_id in corrections_dict:
                #print(f"Processing corrections for record: {record_id}")
                
                for correction in corrections_dict[record_id]:
                    try:
                        # Safely parse the correction using ast.literal_eval
                        corrections_list = ast.literal_eval(correction['correction'])
                        
                        # Validate that corrections_list is a list
                        if not isinstance(corrections_list, list):
                            #print(f"Invalid correction format for {record_id}: {correction}")
                            continue
                        
                        for group_consecutive in corrections_list:
                            # Validate group format
                            if not isinstance(group_consecutive, list) or len(group_consecutive) == 0:
                                #print(f"Invalid group format in {record_id}: {group_consecutive}")
                                continue
                            
                            # Validate each tuple in the group
                            valid_group = True
                            for token_tag_pair in group_consecutive:
                                if not isinstance(token_tag_pair, tuple) or len(token_tag_pair) != 2:
                                    #print(f"Invalid token-tag pair: {token_tag_pair}")
                                    valid_group = False
                                    break
                            
                            if not valid_group:
                                continue
                            
                            num_elements = len(group_consecutive)
                            first_token = group_consecutive[0][0]
                            
                            # Find ALL occurrences of the first token
                            token_indices = [i for i, token in enumerate(corrected_item['tokens']) 
                                           if token == first_token]
                            
                            correction_applied = False
                            
                            for token_index in token_indices:
                                # Check if we have enough tokens remaining
                                if token_index + num_elements > len(corrected_item['tokens']):
                                    continue
                                
                                # Check if the sequence matches
                                is_consecutive = True
                                for i in range(num_elements):
                                    expected_token = group_consecutive[i][0]
                                    actual_token = corrected_item['tokens'][token_index + i]
                                    if actual_token != expected_token:
                                        is_consecutive = False
                                        break
                                
                                if is_consecutive:
                                    # Apply the correction
                                    for i in range(num_elements):
                                        corrected_item['ner_tags'][token_index + i] = group_consecutive[i][1]
                                    
                                    #print(f"Applied correction: {correction.get('comment', 'No comment')} to record {record_id}")
                                    correction_applied = True
                                    break  # Apply only the first matching occurrence
                            
                            #if not correction_applied:
                                #print(f"Could not apply correction for record {record_id}: "
                                             #f"Token sequence {[pair[0] for pair in group_consecutive]} not found")
                    
                    except (ValueError, SyntaxError) as e:
                        #print(f"Error parsing correction for record {record_id}: {e}")
                        #print(f"Problematic correction: {correction}")
                        continue
                    except Exception as e:
                        #print(f"Unexpected error processing record {record_id}: {e}")
                        continue
            
            corrected_dataset.append(corrected_item)
        
        return corrected_dataset
        