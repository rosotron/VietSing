import os

import torch
import numpy as np
from modules.hifigan.hifigan import HifiGanGenerator
from vocoders.hifigan import HifiGAN
from inference.svs.viet.map import viet_word2ph_func

from utils import load_ckpt
from utils.hparams import set_hparams, hparams
from utils.text_encoder import TokenTextEncoder
from pypinyin import pinyin, lazy_pinyin, Style
import librosa
import glob
import re


class BaseSVSInfer:
    def __init__(self, hparams, device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.hparams = hparams
        self.device = device

        # a 
        phone_list = ["a1_T1", "a1_T2", "a1_T3", "a1_T4", "a1_T5", "a1_T6", "a2_T1", "a2_T2", "a2_T3", "a2_T4", "a2_T5", "a2_T6", "a3_T1", "a3_T2", "a3_T3", "a3_T4", "a3_T5", "a3_T6", "ai_T1", "ai_T2", "ai_T3", "ai_T4", "ai_T5", "ai_T6", "ao_T1", "ao_T2", "ao_T3", "ao_T4", "ao_T5", "ao_T6", "ap", "au3_T1", "au3_T2", "au3_T3", "au3_T4", "au3_T5", "au3_T6", "au_T1", "au_T2", "au_T3", "ay3_T1", "ay3_T2", "ay3_T3", "ay3_T4", "ay3_T5", "ay3_T6", "ay_T1", "ay_T2", "ay_T3", "ay_T4", "ay_T5", "ay_T6", "b", "ch", "d1", "d2", "e1_T1", "e1_T2", "e1_T3", "e1_T4", "e1_T5", "e1_T6", "e2_T1", "e2_T2", "e2_T3", "e2_T4", "e2_T5", "e2_T6", "eo_T1", "eo_T2", "eo_T3", "eo_T4", "eo_T5", "eo_T6", "eu_T2", "eu_T3", "eu_T4", "eu_T6", "g", "h", "i_T1", "i_T2", "i_T3", "i_T4", "i_T5", "i_T6", "ie2_T1", "ie2_T2", "ie2_T3", "ie2_T4", "ie2_T5", "ie2_T6", "ieu_T1", "ieu_T2", "ieu_T3", "ieu_T4", "ieu_T6", "iu_T1", "iu_T2", "iu_T3", "iu_T5", "iu_T6", "j", "k", "kh", "l", "m", "n", "ng", "nh", "o1_T1", "o1_T2", "o1_T3", "o1_T4", "o1_T5", "o1_T6", "o2_T1", "o2_T2", "o2_T3", "o2_T4", "o2_T5", "o2_T6", "o3_T1", "o3_T2", "o3_T3", "o3_T4", "o3_T5", "o3_T6", "oa_T1", "oa_T2", "oa_T3", "oa_T4", "oa_T5", "oa_T6", "oe_T1", "oe_T2", "oe_T3", "oe_T4", "oe_T6", "oi2_T1", "oi2_T2", "oi2_T3", "oi2_T4", "oi2_T5", "oi2_T6", "oi3_T1", "oi3_T2", "oi3_T3", "oi3_T4", "oi3_T5", "oi3_T6", "oi_T1", "oi_T2", "oi_T3", "oi_T4", "oi_T5", "oi_T6", "p", "ph", "r", "s", "sil", "sp", "t", "th", "tr", "u1_T1", "u1_T2", "u1_T3", "u1_T4", "u1_T5", "u1_T6", "u2_T1", "u2_T2", "u2_T3", "u2_T4", "u2_T5", "u2_T6", "ua2_T1", "ua2_T2", "ua2_T3", "ua2_T4", "ua2_T5", "ua2_T6", "ua_T1", "ua_T2", "ua_T3", "ua_T4", "ua_T5", "ua_T6", "ui2_T4", "ui_T1", "ui_T2", "ui_T3", "ui_T4", "ui_T5", "ui_T6", "uoi2_T1", "uoi2_T3", "uoi2_T4", "uoi2_T5", "uoi2_T6", "uoi3_T1", "uoi3_T2", "uoi3_T3", "uoi3_T4", "uoi3_T5", "uou_T6", "uu2_T1", "uu2_T2", "uu2_T3", "uu2_T4", "uu2_T5", "uu2_T6", "uy_T1", "uy_T2", "uy_T3", "uy_T4", "uy_T5", "uy_T6", "v", "x"]
        self.ph_encoder = TokenTextEncoder(None, vocab_list=phone_list, replace_oov=',')
        self.pinyin2phs = viet_word2ph_func()
        self.spk_map = {'vietsing': 0}

        self.model = self.build_model()
        self.model.eval()
        self.model.to(self.device)
        self.vocoder = self.build_vocoder()
        self.vocoder.eval()
        self.vocoder.to(self.device)

    def build_model(self):
        raise NotImplementedError

    def forward_model(self, inp):
        raise NotImplementedError

    def build_vocoder(self):
        base_dir = hparams['vocoder_ckpt']
        config_path = f'{base_dir}/config.yaml'
        ckpt = sorted(glob.glob(f'{base_dir}/model_ckpt_steps_*.ckpt'), key=
        lambda x: int(re.findall(f'{base_dir}/model_ckpt_steps_(\d+).ckpt', x)[0]))[-1]
        print('| load HifiGAN: ', ckpt)
        ckpt_dict = torch.load(ckpt, map_location="cpu")
        config = set_hparams(config_path, global_hparams=False)
        state = ckpt_dict["state_dict"]["model_gen"]
        vocoder = HifiGanGenerator(config)
        vocoder.load_state_dict(state, strict=True)
        vocoder.remove_weight_norm()
        vocoder = vocoder.eval().to(self.device)
        return vocoder

    def run_vocoder(self, c, **kwargs):
        c = c.transpose(2, 1)  # [B, 80, T]
        f0 = kwargs.get('f0')  # [B, T]
        if f0 is not None and hparams.get('use_nsf'):
            # f0 = torch.FloatTensor(f0).to(self.device)
            y = self.vocoder(c, f0).view(-1)
        else:
            y = self.vocoder(c).view(-1)
            # [T]
        return y[None]

    def preprocess_word_level_input(self, inp):
        text_raw = inp['text'].split(' ')

        # lyric
        ph_per_word_lst = [self.pinyin2phs[word.strip()] for word in text_raw if word.strip() in self.pinyin2phs]
        print(ph_per_word_lst) 

        # Note
        note_per_word_lst = [x.strip() for x in inp['notes'].split('|') if x.strip() != '']
        mididur_per_word_lst = [x.strip() for x in inp['notes_duration'].split('|') if x.strip() != '']

        if len(note_per_word_lst) == len(ph_per_word_lst) == len(mididur_per_word_lst):
            print('Pass word-notes check.')
        else:
            print('The number of words does\'t match the number of notes\' windows. ',
                  'You should split the note(s) for each word by | mark.')
            print(ph_per_word_lst, note_per_word_lst, mididur_per_word_lst)
            print(len(ph_per_word_lst), len(note_per_word_lst), len(mididur_per_word_lst))
            return None

        note_lst = []
        ph_lst = []
        midi_dur_lst = []
        is_slur = []
        for idx, ph_per_word in enumerate(ph_per_word_lst):
            # for phs in one word:
            # single ph like ['ai']  or multiple phs like ['n', 'i']
            ph_in_this_word = ph_per_word.split()

            # for notes in one word:
            # single note like ['D4'] or multiple notes like ['D4', 'E4'] which means a 'slur' here.
            note_in_this_word = note_per_word_lst[idx].split()
            midi_dur_in_this_word = mididur_per_word_lst[idx].split()
            # process for the model input
            # Step 1.
            #  Deal with note of 'not slur' case or the first note of 'slur' case
            #  j        ie
            #  F#4/Gb4  F#4/Gb4
            #  0        0
            for ph in ph_in_this_word:
                ph_lst.append(ph)
                note_lst.append(note_in_this_word[0])
                midi_dur_lst.append(midi_dur_in_this_word[0])
                is_slur.append(0)
            # step 2.
            #  Deal with the 2nd, 3rd... notes of 'slur' case
            #  j        ie         ie
            #  F#4/Gb4  F#4/Gb4    C#4/Db4
            #  0        0          1
            if len(note_in_this_word) > 1:  # is_slur = True, we should repeat the YUNMU to match the 2nd, 3rd... notes.
                for idx in range(1, len(note_in_this_word)):
                    ph_lst.append(ph_in_this_word[-1])
                    note_lst.append(note_in_this_word[idx])
                    midi_dur_lst.append(midi_dur_in_this_word[idx])
                    is_slur.append(1)
        ph_seq = ' '.join(ph_lst)

        if len(ph_lst) == len(note_lst) == len(midi_dur_lst):
            print(len(ph_lst), len(note_lst), len(midi_dur_lst))
            print('Pass word-notes check.')
        else:
            print('The number of words does\'t match the number of notes\' windows. ',
                  'You should split the note(s) for each word by | mark.')
            return None
        return ph_seq, note_lst, midi_dur_lst, is_slur

    def preprocess_phoneme_level_input(self, inp):
        ph_seq = inp['ph_seq']
        note_lst = inp['note_seq'].split()
        midi_dur_lst = inp['note_dur_seq'].split()
        is_slur = [float(x) for x in inp['is_slur_seq'].split()]
        print(len(note_lst), len(ph_seq.split()), len(midi_dur_lst))
        if len(note_lst) == len(ph_seq.split()) == len(midi_dur_lst):
            print('Pass word-notes check.')
        else:
            print('The number of words does\'t match the number of notes\' windows. ',
                  'You should split the note(s) for each word by | mark.')
            return None
        return ph_seq, note_lst, midi_dur_lst, is_slur

    def preprocess_input(self, inp, input_type='word'):
        """

        :param inp: {'text': str, 'item_name': (str, optional), 'spk_name': (str, optional)}
        :return:
        """

        item_name = inp.get('item_name', '<ITEM_NAME>')
        spk_name = inp.get('spk_name', 'vietsing')

        # single spk
        spk_id = self.spk_map[spk_name]

        # get ph seq, note lst, midi dur lst, is slur lst.
        if input_type == 'word':
            ret = self.preprocess_word_level_input(inp) 
        elif input_type == 'phoneme':  # like transcriptions.txt in dataset.
            ret = self.preprocess_phoneme_level_input(inp) 
        else:
            print('Invalid input type.')
            return None

        if ret:
            ph_seq, note_lst, midi_dur_lst, is_slur = ret
        else:
            print('==========> Preprocess_word_level or phone_level input wrong.')
            return None

        # convert note lst to midi id; convert note dur lst to midi duration
        try:
            # midis = [librosa.note_to_midi(x.split("/")[0]) if x != 'rest' else 0
            #          for x in note_lst]
            midis = [int(float(x)) for x in note_lst]  # Since basic pitch outputs the MIDI number instead of note.
            midi_dur_lst = [float(x) for x in midi_dur_lst]
        except Exception as e:
            print(e)
            print('Invalid Input Type.')
            return None

        ph_token = self.ph_encoder.encode(ph_seq)
        item = {'item_name': item_name, 'text': inp['text'], 'ph': ph_seq, 'spk_id': spk_id,
                'ph_token': ph_token, 'pitch_midi': np.asarray(midis), 'midi_dur': np.asarray(midi_dur_lst),
                'is_slur': np.asarray(is_slur), }
        item['ph_len'] = len(item['ph_token'])
        return item

    def input_to_batch(self, item):
        item_names = [item['item_name']]
        text = [item['text']]
        ph = [item['ph']]
        txt_tokens = torch.LongTensor(item['ph_token'])[None, :].to(self.device)
        txt_lengths = torch.LongTensor([txt_tokens.shape[1]]).to(self.device)
        spk_ids = torch.LongTensor(item['spk_id'])[None, :].to(self.device)

        pitch_midi = torch.LongTensor(item['pitch_midi'])[None, :hparams['max_frames']].to(self.device)
        midi_dur = torch.FloatTensor(item['midi_dur'])[None, :hparams['max_frames']].to(self.device)
        is_slur = torch.LongTensor(item['is_slur'])[None, :hparams['max_frames']].to(self.device)

        batch = {
            'item_name': item_names,
            'text': text,
            'ph': ph,
            'txt_tokens': txt_tokens,
            'txt_lengths': txt_lengths,
            'spk_ids': spk_ids,
            'pitch_midi': pitch_midi,
            'midi_dur': midi_dur,
            'is_slur': is_slur
        }
        return batch

    def postprocess_output(self, output):
        return output

    def infer_once(self, inp):
        inp = self.preprocess_input(inp, input_type=inp['input_type'] if inp.get('input_type') else 'word')
        output = self.forward_model(inp)
        output = self.postprocess_output(output)
        return output

    @classmethod
    def example_run(cls, inp, wav_name):
        from utils.audio import save_wav
        set_hparams(print_hparams=False)
        # print(cls)
        infer_ins = cls(hparams)
        out = infer_ins.infer_once(inp) 
        os.makedirs('infer_out', exist_ok=True)
        # Rename your inference wav
        save_wav(out, f'infer_out/{wav_name}.wav', hparams['audio_sample_rate'])
