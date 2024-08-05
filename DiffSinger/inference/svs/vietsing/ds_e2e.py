import os
import torch
# from inference.tts.fs import FastSpeechInfer
# from modules.tts.fs2_orig import FastSpeech2Orig
from inference.svs.vietsing.base_svs_infer import BaseSVSInfer
from utils import load_ckpt
from utils.hparams import hparams
from usr.diff.shallow_diffusion_tts import GaussianDiffusion
from usr.diffsinger_task import DIFF_DECODERS
from modules.fastspeech.pe import PitchExtractor
import utils


class DiffSingerE2EInfer(BaseSVSInfer):
    def build_model(self):
        model = GaussianDiffusion(
            phone_encoder=self.ph_encoder,
            out_dims=hparams['audio_num_mel_bins'], denoise_fn=DIFF_DECODERS[hparams['diff_decoder_type']](hparams),
            timesteps=hparams['timesteps'],
            K_step=hparams['K_step'],
            loss_type=hparams['diff_loss_type'],
            spec_min=hparams['spec_min'], spec_max=hparams['spec_max'],
        )
        model.eval()
        load_ckpt(model, hparams['work_dir'], 'model')

        if hparams.get('pe_enable') is not None and hparams['pe_enable']:
            self.pe = PitchExtractor().to(self.device)
            utils.load_ckpt(self.pe, hparams['pe_ckpt'], 'model', strict=True)
            self.pe.eval()
        return model

    def forward_model(self, inp):
        sample = self.input_to_batch(inp)
        txt_tokens = sample['txt_tokens']  # [B, T_t]
        spk_id = sample.get('spk_ids')
        with torch.no_grad():
            output = self.model(txt_tokens, spk_id=spk_id, ref_mels=None, infer=True,
                                pitch_midi=sample['pitch_midi'], midi_dur=sample['midi_dur'],
                                is_slur=sample['is_slur'])
            mel_out = output['mel_out']  # [B, T,80]
            if hparams.get('pe_enable') is not None and hparams['pe_enable']:
                f0_pred = self.pe(mel_out)['f0_denorm_pred']  # pe predict from Pred mel
            else:
                f0_pred = output['f0_denorm']
            wav_out = self.run_vocoder(mel_out, f0=f0_pred)
        wav_out = wav_out.cpu().numpy()
        return wav_out[0]

if __name__ == '__main__':

    inp = {'input_type': 'phoneme',
        'spk_name': 'vietsing'}
    raw_folder = './data/raw/ha_trang'
    inp['text'] = open(os.path.join(raw_folder, 'word_ha_trang.txt')).readline()
    raw_folder = './data/raw/ha_trang/omni'
    inp['ph_seq'] = open(os.path.join(raw_folder, 'ph_parody.txt')).readline()
    inp['note_seq'] = open(os.path.join(raw_folder, 'midi_ha_trang.txt')).readline()
    inp['note_dur_seq'] = open(os.path.join(raw_folder, 'mididur_ha_trang.txt')).readline()
    inp['is_slur_seq'] = open(os.path.join(raw_folder, 'slur_ha_trang.txt')).readline()
        
    DiffSingerE2EInfer.example_run(inp, 'parody_new')