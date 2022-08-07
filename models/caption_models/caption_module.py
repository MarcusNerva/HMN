import torch
import torch.nn as nn


class CaptionModule(nn.Module):
    """
    CaptionModule and its child classes are complementary.
    """
    def __init__(self, beam_size):
        super(CaptionModule, self).__init__()
        self.beam_size = beam_size

    def __beam_step(self, t, logprobs, beam_seq, beam_seq_logprobs, beam_logprobs_sum, state):
        beam_size = self.beam_size

        probs, idx = torch.sort(logprobs, dim=1, descending=True)
        candidates = []
        rows = beam_size if t >= 1 else 1
        cols = min(beam_size, probs.size(1))

        for r in range(rows):
            for c in range(cols):
                tmp_logprob = probs[r, c]
                tmp_sum = beam_logprobs_sum[r] + tmp_logprob
                tmp_idx = idx[r, c]
                candidates.append({'sum': tmp_sum, 'logprob': tmp_logprob, 'ix': tmp_idx, 'beam': r})

        candidates = sorted(candidates, key=lambda x: -x['sum'])
        prev_seq = beam_seq[:, :t].clone()
        prev_seq_probs = beam_seq_logprobs[:, :t].clone()
        prev_logprobs_sum = beam_logprobs_sum.clone()
        new_state = [_.clone() for _ in state]

        for i in range(beam_size):
            candidate_i = candidates[i]
            beam = candidate_i['beam']
            ix = candidate_i['ix']
            logprob = candidate_i['logprob']

            beam_seq[i, :t] = prev_seq[beam, :]
            beam_seq_logprobs[i, :t] = prev_seq_probs[beam, :]
            beam_seq[i, t] = ix
            beam_seq_logprobs[i, t] = logprob
            beam_logprobs_sum[i] = prev_logprobs_sum[beam] + logprob
            for j in range(len(new_state)):
                new_state[j][:, i, :] = state[j][:, beam, :]

        return beam_seq, beam_seq_logprobs, beam_logprobs_sum, new_state

    def __beam_search(self, objects_feats, action_feats, caption_feats, objects_pending, action_pending, caption_pending, state):
        beam_size = self.beam_size
        device = caption_feats.device if caption_feats is not None else objects_feats.device

        beam_seq = torch.LongTensor(beam_size, self.max_caption_len).fill_(self.eos_idx)
        beam_seq_logprobs = torch.FloatTensor(beam_size, self.max_caption_len).zero_()
        beam_logprobs_sum = torch.zeros(beam_size)
        ret = []

        it = torch.LongTensor(beam_size).fill_(self.sos_idx).to(device)
        it_embed = self.embedding(it)
        output_prob, state = self.forward_decoder(objects_feats, action_feats, caption_feats, objects_pending, action_pending, caption_pending, it_embed, state)
        logprob = output_prob

        for t in range(self.max_caption_len):
            # suppress UNK tokens in the decoding. So the probs of 'UNK' are extremely low
            logprob[:, self.unk_idx] = logprob[:, self.unk_idx] - 1000.0
            beam_seq, beam_seq_logprobs, beam_logprobs_sum, state = self.__beam_step(t=t,
                                                                                   logprobs=logprob,
                                                                                   beam_seq=beam_seq,
                                                                                   beam_seq_logprobs=beam_seq_logprobs,
                                                                                   beam_logprobs_sum=beam_logprobs_sum,
                                                                                   state=state)

            for j in range(beam_size):
                if beam_seq[j, t] == self.eos_idx or t == self.max_caption_len - 1:
                    final_beam = {
                        'seq': beam_seq[j, :].clone(),
                        'seq_logprob': beam_seq_logprobs[j, :].clone(),
                        'sum_logprob': beam_logprobs_sum[j].clone()
                    }
                    ret.append(final_beam)
                    beam_logprobs_sum[j] = -1000.0

            it = beam_seq[:, t].to(device)
            it_embed = self.embedding(it).to(device)
            output_prob, state = self.forward_decoder(objects_feats, action_feats, caption_feats, objects_pending, action_pending, caption_pending, it_embed, state)
            logprob = output_prob

        ret = sorted(ret, key=lambda x: -x['sum_logprob'])[:beam_size]
        return ret

    def sample_beam(self, objects_feats, vps_feats, caption_feats, objects_semantics, action_semantics, caption_semantics):
        beam_size = self.beam_size
        batch_size = caption_feats.shape[0] if caption_feats is not None else objects_feats.shape[0]
        hidden_dim = caption_feats.shape[-1] if caption_feats is not None else objects_feats.shape[-1]
        device = caption_feats.device if caption_feats is not None else objects_feats.device

        seq = torch.LongTensor(batch_size, self.max_caption_len).fill_(self.eos_idx)
        seq_probabilities = torch.FloatTensor(batch_size, self.max_caption_len)
        done_beam = [[] for _ in range(batch_size)]

        for i in range(batch_size):
            single_objects_feats = objects_feats[i, ...][None, ...] if objects_feats is not None else None  # (1, sample_numb, obj_per_frame, hidden_dim)
            single_vps_feats = vps_feats[i, ...][None, ...] if vps_feats is not None else None  # (1, sample_numb, hidden_dim)
            single_caption_feats = caption_feats[i, ...][None, ...] if caption_feats is not None else None  # (1, sample_numb, hidden_dim)
            single_objects_semantics = objects_semantics[i, ...][None, ...] if objects_semantics is not None else None  # (1, max_objects, word_dim)
            single_action_semantics = action_semantics[i, ...][None, ...] if action_semantics is not None else None  # (1, semantics_dim)
            single_caption_semantics = caption_semantics[i, ...][None, ...] if caption_semantics is not None else None  # (1, semantics_dim)
            # print('====={}'.format(single_objects_semantics.shape))

            single_objects_feats = single_objects_feats.repeat(beam_size, 1, 1) if single_objects_feats is not None else None  # (beam_size, max_objects, hidden_dim)
            single_vps_feats = single_vps_feats.repeat(beam_size, 1, 1) if single_vps_feats is not None else None  # (beam_size, sample_numb, hidden_dim)
            single_caption_feats = single_caption_feats.repeat(beam_size, 1, 1) if single_caption_feats is not None else None  # (beam_size, sample_numb, hidden_dim)
            single_objects_semantics = single_objects_semantics.repeat(beam_size, 1, 1) if single_objects_semantics is not None else None  # (beam_size, max_objects, word_dim)
            single_action_semantics = single_action_semantics.repeat(beam_size, 1) if single_action_semantics is not None else None  # (beam_size, semantics_dim)
            single_caption_semantics = single_caption_semantics.repeat(beam_size, 1) if single_caption_semantics is not None else None  # (beam_size, semantics_dim)

            state = self.get_rnn_init_hidden(beam_size, hidden_dim, device)

            done_beam[i] = self.__beam_search(single_objects_feats, single_vps_feats,
                                              single_caption_feats, single_objects_semantics,
                                              single_action_semantics, single_caption_semantics,
                                              state)
            seq[i, ...] = done_beam[i][0]['seq']
            seq_probabilities[i, ...] = done_beam[i][0]['seq_logprob']

        return seq, seq_probabilities

    def sample(self, objects, object_masks, feature2ds, feature3ds, is_sample_max=True):
        beam_size = self.beam_size
        temperature = self.temperature
        batch_size = feature2ds.shape[0]
        device = feature2ds.device

        objects_feats, action_feats, caption_feats, \
        objects_semantics, action_semantics, caption_semantics = self.forward_encoder(objects, object_masks, feature2ds, feature3ds)

        if beam_size > 1:
            return self.sample_beam(objects_feats, action_feats, caption_feats,
                                    objects_semantics, action_semantics, caption_semantics)

        state = self.get_rnn_init_hidden(batch_size, device)
        seq, seq_probabilities = [], []

        for t in range(self.max_caption_len):
            if t == 0:
                it = objects_feats.new(batch_size).fill_(self.sos_idx).long()
            elif is_sample_max:
                sampleLogprobs, it = torch.max(log_probabilities.detach(), 1)
                it = it.view(-1).long()
            else:
                prev_probabilities = torch.exp(torch.div(log_probabilities.detach(), temperature))
                it = torch.multinomial(prev_probabilities, 1)
                sampleLogprobs = log_probabilities.gather(1, it)
                it = it.view(-1).long()

            it_embed = self.embedding(it)

            if t >= 1:
                if t == 1:
                    unfinished = it > 0
                else:
                    unfinished = unfinished * (it > 0)
                # if unfinished.sum() == 0: break
                it = it * unfinished.type_as(it)
                seq.append(it)
                seq_probabilities.append(sampleLogprobs.view(-1))

            it_embed = it_embed.to(device)
            log_probabilities, state = self.forward_decoder(objects_feats, action_feats, caption_feats,
                                                            objects_semantics, action_semantics,
                                                            caption_semantics, it_embed, state)

        seq.append(it.new(batch_size).long().fill_(self.eos_idx))
        seq_probabilities.append(sampleLogprobs.view(-1))
        return torch.cat([_.unsqueeze(1) for _ in seq], 1), torch.cat([_.unsqueeze(1) for _ in seq_probabilities], 1)
