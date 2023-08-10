import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import *
import streamlit as st
import pandas as pd
import os
from pathlib import Path
import spacy
from spacy import displacy
from torch.utils.data import DataLoader
import warnings,transformers,logging,torch
from transformers import *
from transformers import AutoModelForSequenceClassification,AutoTokenizer
import datasets
from datasets import load_dataset, Dataset, DatasetDict
from sklearn.metrics import log_loss
import torch.nn.functional as F
import spacy_streamlit

custom_css = """
<style>
.dynamic-html {
    font-size: 20px;
    line-height: 1.5;
    overflow: hidden;
    white-space: nowrap;
    text-overflow: ellipsis;
}
</style>
"""


st.set_page_config(layout='wide')
st.markdown(custom_css, unsafe_allow_html=True)


final_predictons = pd.DataFrame()
MAX_LEN = 1024

st.markdown("<h1><em>ArguSense: Intelligent NLP-driven Argument Evaluation and Effectiveness Analysis</em></h1>", unsafe_allow_html=True)

st.subheader("Enter your argument in the textbox below")

if "tokenizer" not in st.session_state:
    st.session_state['tokenizer'] = None

if "model" not in st.session_state:
    st.session_state['model'] = None

if "loaded_tokz" not in st.session_state:
    st.session_state['loaded_tokz'] = None

if "loaded_trainer" not in st.session_state:
    st.session_state['loaded_trainer'] = None

if "button" not in st.session_state:
    st.session_state['button'] = None

# Add a textbox to the app
user_input = st.text_area("Enter your text:", "Default text", height=400)


spinner_text = "Setting up your environment..."

## model building start
@st.cache_data(show_spinner = spinner_text)
def model_building():
    st.session_state['tokenizer'] = AutoTokenizer.from_pretrained('input')

    targets = np.load('targets_1024.npy')
    train_tokens = np.load('tokens_1024.npy')
    train_attention = np.load('attention_1024.npy')

    def build_model():
        tokens = tf.keras.layers.Input(shape=(MAX_LEN,), name='tokens', dtype=tf.int32)
        attention = tf.keras.layers.Input(shape=(MAX_LEN,), name='attention', dtype=tf.int32)

        config = AutoConfig.from_pretrained('config.json')
        backbone = TFAutoModel.from_pretrained('tf_model.h5', config=config)

        x = backbone(tokens, attention_mask=attention)
        x = tf.keras.layers.Dense(256, activation='relu')(x[0])
        x = tf.keras.layers.Dense(15, activation='softmax', dtype='float32')(x)

        st.session_state['model'] = tf.keras.Model(inputs=[tokens, attention], outputs=x)
        st.session_state['model'].compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4),
                    loss=[tf.keras.losses.CategoricalCrossentropy()],
                    metrics=[tf.keras.metrics.CategoricalAccuracy()])

        return st.session_state['model']

    tf.keras.utils.get_custom_objects()["swish"] = tf.keras.activations.swish
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    st.session_state['model'] = build_model()

    st.session_state['model'].load_weights('long_v14.h5')

    model_nm = 'deberta'

    # Load the trained model and tokenizer
    loaded_model = AutoModelForSequenceClassification.from_pretrained(model_nm)
    st.session_state['loaded_tokz'] = AutoTokenizer.from_pretrained(model_nm)

    # Set up the Trainer for prediction
    st.session_state['loaded_trainer'] = Trainer(model=loaded_model, tokenizer=st.session_state['loaded_tokz'], compute_metrics=score)

    st.success("Your Workspace is ready!")




def score(preds): return {'log loss': log_loss(preds.label_ids, F.softmax(torch.Tensor(preds.predictions)))}


model_building()
st.session_state['button'] =  st.button("Proceed")

## Model Build complete

if st.session_state['button'] and len(user_input)>0:

    with st.spinner("Evaluating your argument...."):
        # CONVERT TEST TEXT TO TOKENS
        test_tokens = np.zeros((1, MAX_LEN), dtype='int32')
        test_attention = np.zeros((1, MAX_LEN), dtype='int32')

        # READ TRAIN TEXT, TOKENIZE, AND SAVE IN TOKEN ARRAYS
        n = "0FB0700DAF44"

        text_to_write = user_input

        txt = text_to_write
        tokens = st.session_state['tokenizer'].encode_plus(txt, max_length=MAX_LEN, padding='max_length',
                                    truncation=True, return_offsets_mapping=True)
        test_tokens[0,] = tokens['input_ids']
        test_attention[0,] = tokens['attention_mask']

        # INFER TEST TEXTS
        p = st.session_state['model'].predict([test_tokens, test_attention],
                        batch_size=16, verbose=2)
        test_preds = np.argmax(p, axis=-1)


        target_map_rev = {0: 'Lead', 1: 'Position', 2: 'Evidence', 3: 'Claim', 4: 'Concluding Statement',
                        5: 'Counterclaim', 6: 'Rebuttal', 7: 'blank'}

        all_predictions = []

        tokens = st.session_state['tokenizer'].encode_plus(txt, max_length=MAX_LEN, padding='max_length',
                                    truncation=True, return_offsets_mapping=True)
        off = tokens['offset_mapping']

        # GET WORD POSITIONS IN CHARS
        w = []
        blank = True
        for i in range(len(txt)):
            if (txt[i] != ' ') & (txt[i] != '\n') & (txt[i] != '\xa0') & (txt[i] != '\x85') & (blank == True):
                w.append(i)
                blank = False
            elif (txt[i] == ' ') | (txt[i] == '\n') | (txt[i] == '\xa0') | (txt[i] == '\x85'):
                blank = True

        w.append(1e6)

        # MAPPING FROM TOKENS TO WORDS
        word_map = -1 * np.ones(MAX_LEN, dtype='int32')
        w_i = 0

        for i in range(len(off)):
            if off[i][1] == 0: continue
            while off[i][0] >= w[w_i + 1]: w_i += 1
            word_map[i] = int(w_i)

        pred = test_preds[0,] / 2.0

        i = 0
        while i < MAX_LEN:
            prediction = []
            start = pred[i]
            if start in [0, 1, 2, 3, 4, 5, 6, 7]:
                prediction.append(word_map[i])
                i += 1
                if i >= MAX_LEN: break
                while pred[i] == start + 0.5:
                    if not word_map[i] in prediction:
                        prediction.append(word_map[i])
                    i += 1
                    if i >= MAX_LEN: break
            else:
                i += 1
            prediction = [x for x in prediction if x != -1]
            if len(prediction) > 4:
                all_predictions.append((n, target_map_rev[int(start)],
                                        ' '.join([str(x) for x in prediction])))
                
        
        # MAKE DATAFRAME
        df = pd.DataFrame(all_predictions)

        final_predictons = df

    if len(final_predictons) > 0:

        df.columns = ['id', 'discourse_type', 'predictionstring']

        id_num = df['id']
        counter = 0
        last_char_index = -1
        discourse_start = []
        discourse_end = []
        discourse_text = []
        discourse_id = []

        for ids in id_num:
            first_prediction_string = df['predictionstring'].iloc[counter]

            text_no = first_prediction_string.split(' ')
            text_start = int(text_no[0])
            text_end = int(text_no[-1])

            words = txt.split()

            text = words[text_start:text_end + 1]
            text2 = words[0:text_start]
            # print(text2)

            current_word_char = len(" ".join(text2))
            # print("current_word_char: ", current_word_char)
            if current_word_char != 0:
                current_word_char += 1
            start_char_index = 0 + current_word_char

            total_chars = len(" ".join(text))

            last_char_index = start_char_index + total_chars
            # print("start index : ", start_char_index)
            # print("end index : ", last_char_index)

            discourse_text.append(" ".join(text))
            discourse_start.append(start_char_index)
            discourse_end.append(last_char_index)
            discourse_id.append(counter)
            counter += 1

        df['discourse_start'] = discourse_start
        df['discourse_end'] = discourse_end
        df['discourse_text'] = discourse_text
        df['discourse_id'] = discourse_id

        sep = st.session_state['loaded_tokz'].sep_token

        def tok_func(x): return st.session_state['loaded_tokz'](x["inputs"], truncation=True)

        df['inputs'] = df.discourse_type + sep + df.discourse_text

        def get_dds(df, train=True):
            ds = Dataset.from_pandas(df)
            to_remove = ['discourse_text','discourse_type','inputs','discourse_id','id','predictionstring','discourse_start','discourse_end']
            tok_ds = ds.map(tok_func, batched=True, remove_columns=to_remove)
            return tok_ds

        loaded_test_ds = get_dds(df, train=False)
        loaded_preds = F.softmax(torch.Tensor(st.session_state['loaded_trainer'].predict(loaded_test_ds).predictions)).numpy().astype(float)


        final_df = pd.DataFrame()
        final_df['id'] = df['id']
        final_df['discourse_type'] = df['discourse_type']
        final_df['discourse_start'] = df['discourse_start']
        final_df['discourse_end'] = df['discourse_end']
        final_df['Ineffective'] = loaded_preds[:,0]
        final_df['Adequate'] = loaded_preds[:,1]
        final_df['Effective'] = loaded_preds[:,2]

        # Function to determine the highest effectiveness level
        def get_effectiveness(row):
            max_effectiveness = max(row['Ineffective'], row['Adequate'], row['Effective'])
            if row['Ineffective'] == max_effectiveness:
                return f"{row['discourse_type']} - Ineffective"
            elif row['Adequate'] == max_effectiveness:
                return f"{row['discourse_type']} - Adequate"
            else:
                return f"{row['discourse_type']} - Effective"

        # Apply the function to create the "Effectiveness" column
        final_df['Effectiveness'] = final_df.apply(get_effectiveness, axis=1)

        colors = {
                    'Lead - Ineffective': '#8000ff',
                    'Lead - Adequate': '#8000ff',
                    'Lead - Effective': '#8000ff',
                    'Position - Ineffective': '#2b7ff6',
                    'Position - Adequate': '#2b7ff6',
                    'Position - Effective': '#2b7ff6',
                    'Evidence - Ineffective': '#2adddd',
                    'Evidence - Adequate': '#2adddd',
                    'Evidence - Effective': '#2adddd',
                    'Claim - Ineffective': '#80ffb4',
                    'Claim - Adequate': '#80ffb4',
                    'Claim - Effective': '#80ffb4',
                    'Concluding Statement - Ineffective': '#d4dd80',
                    'Concluding Statement - Adequate': '#d4dd80',
                    'Concluding Statement - Effective': '#d4dd80',
                    'Counterclaim - Ineffective': '#ff8042',
                    'Counterclaim - Adequate': '#ff8042',
                    'Counterclaim - Effective': '#ff8042',
                    'Rebuttal - Ineffective': '#ff0000',
                    'Rebuttal - Adequate': '#ff0000',
                    'Rebuttal - Effective': '#ff0000'
                }

        def visualize(example):
            ents = []
            for i, row in final_df[final_df['id'] == example].iterrows():
                ents.append({
                                'start': int(row['discourse_start']),
                                'end': int(row['discourse_end']),
                                'label': row['Effectiveness']
                            })

            #with open(path/f'{example}.txt', 'r') as file: data = file.read()

            data = txt
            doc2 = {
                "text": data,
                "ents": ents,
                "title": "Evaluated Argument"
            }

            options = {"ents": final_df.Effectiveness.unique().tolist(), "colors": colors}
            html = displacy.render(doc2, style="ent", options=options, manual = True)
            
            # Set the class of the HTML content
            styled_html = f'<div class="dynamic-html">{html}</div>'
            st.write(styled_html, unsafe_allow_html=True)



        examples = final_df['id'].values.tolist()

        # Create a set to keep track of visualized IDs
        visualized_ids = set()

        for ex in examples:
            if ex not in visualized_ids:
                visualize(ex)
                print('\n')
                visualized_ids.add(ex)

