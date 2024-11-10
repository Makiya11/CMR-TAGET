import warnings
import torch
from torch.nn import functional as F
from layers.decoder import top_k_top_p_filtering

class AutoRegressiveBeamSearch(object):
    def __init__(
        self,
        eos_index: int,
        max_steps: int = 50,
        beam_size: int = 5,
        per_node_beam_size: int = 2,
        fix_missing_prefix=False,
    ) -> None:
        self._eos_index = eos_index
        self.max_steps = max_steps
        self.beam_size = beam_size
        self.per_node_beam_size = per_node_beam_size or beam_size
        self.fix_missing_prefix = fix_missing_prefix
        assert fix_missing_prefix, 'should always true'

    def search(self, start_predictions, step,
               only_return_best=True,
               do_sample=False,
               top_k=0,
               top_p=None,
               num_return_sequences=1,
               temperature=1,
               ):
        if num_return_sequences > 1:
            start_predictions = start_predictions[:, None, :].expand(
                start_predictions.shape[0],
                num_return_sequences,
                start_predictions.shape[1])
            start_predictions = start_predictions.reshape(-1, start_predictions.shape[-1])

        batch_size = start_predictions.size()[0]
        if not self.fix_missing_prefix:
            # List of `(batch_size, beam_size, length)` tensors.
            # Does not include the start symbols, which are implicit.
            predictions: torch.Tensor = torch.empty(
                (batch_size, self.beam_size, 0),
                dtype=torch.long, device=start_predictions.device
            )
        else:
            #predictions = start_predictions.unsqueeze(-1).expand((batch_size, self.beam_size, start_predictions.shape[-1]))
            predictions = start_predictions.unsqueeze(1).expand((batch_size, self.beam_size, start_predictions.shape[-1]))
        # Calculate the first timestep. This is done outside the main loop
        # because we are going from a single decoder input (the output from the
        # encoder) to the top `beam_size` decoder outputs. On the other hand,
        # within the main loop we are going from the `beam_size` elements of the
        # beam to `beam_size`^2 candidates from which we will select the top
        # `beam_size` elements for the next iteration.
        # shape: (batch_size, num_classes)
        start_class_logits = step(start_predictions)

        if temperature != 1:
            assert do_sample
            start_class_logits = start_class_logits / temperature

        # Convert logits to logprobs.
        # shape: (batch_size * beam_size, vocab_size)
        start_class_logprobs = F.log_softmax(start_class_logits, dim=1)

        num_classes = start_class_logprobs.size()[1]

        if not do_sample:
            # shape: (batch_size, beam_size), (batch_size, beam_size)
            start_top_logprobs, start_predicted_classes = start_class_logprobs.topk(
                self.beam_size
            )
        else:
            start_predicted_classes = torch.multinomial(start_class_logits.softmax(dim=1),
                    num_samples=self.beam_size)  # (batch_size, num_beams)
            start_top_logprobs = torch.gather(start_class_logprobs, -1, start_predicted_classes)  # (batch_size, num_beams)

        if (
            self.beam_size == 1
            and (start_predicted_classes == self._eos_index).all()
        ):
            warnings.warn(
                "Empty captions predicted. You may want to increase beam "
                "size or ensure your step function is working properly.",
                RuntimeWarning,
            )
            if only_return_best:
                return start_predicted_classes, start_top_logprobs
            else:
                return start_predicted_classes.unsqueeze(-1), start_top_logprobs

        # The log probs for the last time step.
        # shape: (batch_size, beam_size)
        last_logprobs = start_top_logprobs

        # shape: (batch_size, beam_size, sequence_length)
        predictions = torch.cat([predictions, start_predicted_classes.unsqueeze(-1)], dim=-1)

        # Log probability tensor that mandates that the end token is selected.
        # shape: (batch_size * beam_size, num_classes)
        logprobs_after_end = start_class_logprobs.new_full(
            (batch_size * self.beam_size, num_classes), float("-inf")
        )
        logprobs_after_end[:, self._eos_index] = 0.0

        logits_after_end = start_class_logprobs.new_full(
            (batch_size * self.beam_size, num_classes), float("-inf")
        )
        logits_after_end[:, self._eos_index] = 0

        #for timestep in range(self.max_steps - 1):
        while predictions.shape[-1] < self.max_steps:
            # shape: (batch_size * beam_size,)
            last_predictions = predictions[:, :, -1].reshape(batch_size * self.beam_size)

            # If every predicted token from the last step is `self._eos_index`,
            # then we can stop early.
            if (last_predictions == self._eos_index).all():
                break

            predictions_so_far = predictions.view(
                batch_size * self.beam_size, -1
            )
            # shape: (batch_size * beam_size, num_classes)
            class_logits = step(predictions_so_far)

            # Set logprobs of last predicted tokens as high negative value to avoid
            # repetition in caption.
            class_logits = class_logits.scatter(1, predictions_so_far[:, -1].view((-1, 1)), -10000)

            # shape: (batch_size * beam_size, num_classes)
            last_predictions_expanded = last_predictions.unsqueeze(-1).expand(
                batch_size * self.beam_size, num_classes
            )

            # Here we are finding any beams where we predicted the end token in
            # the previous timestep and replacing the distribution with a
            # one-hot distribution, forcing the beam to predict the end token
            # this timestep as well.
            # shape: (batch_size * beam_size, num_classes)
            #cleaned_logprobs = torch.where(
                #last_predictions_expanded == self._eos_index,
                #logprobs_after_end,
                #class_logprobs,
            #)
            class_logits = torch.where(
                last_predictions_expanded == self._eos_index,
                logits_after_end,
                class_logits,
            )

            # Convert logits to logprobs.
            # shape: (batch_size * beam_size, vocab_size)
            #for index in range(batch_size * self.beam_size):
                ##class_logprobs[index, predictions_so_far[index, -1]] = -10000
                #class_logprobs[index, predictions_so_far[index, -1]] = -10000
            class_logprobs = F.log_softmax(class_logits, dim=1)

            # Set logprobs of last predicted tokens as high negative value to avoid
            # repetition in caption.
            #class_logprobs = class_logprobs.scatter(1, predictions_so_far[:, -1].view((-1, 1)), -10000)

            if not do_sample:
                # shape (both): (batch_size * beam_size, per_node_beam_size)
                top_logprobs, predicted_classes = class_logprobs.topk(
                    self.per_node_beam_size
                )
            else:
                if temperature != 1:
                    class_logits = class_logits / temperature
                #class_logits = top_k_top_p_filtering(class_logits, top_k=top_k, top_p=top_p)
                predicted_classes = torch.multinomial(class_logits.softmax(dim=1),
                        num_samples=self.per_node_beam_size)  # (batch_size * num_beams, TOPN_PER_BEAM)
                top_logprobs = torch.gather(class_logprobs, -1, predicted_classes)  # (batch_size * num_beams, per_node_beam_size)

            # Here we expand the last log probs to `(batch_size * beam_size,
            # per_node_beam_size)` so that we can add them to the current log
            # probs for this timestep. This lets us maintain the log
            # probability of each element on the beam.
            # shape: (batch_size * beam_size, per_node_beam_size)
            expanded_last_logprobs = (
                last_logprobs.unsqueeze(2)
                .expand(batch_size, self.beam_size, self.per_node_beam_size)
                .reshape(batch_size * self.beam_size, self.per_node_beam_size)
            )
            # shape: (batch_size * beam_size, per_node_beam_size)
            summed_top_logprobs = top_logprobs + expanded_last_logprobs

            # shape: (batch_size, beam_size * per_node_beam_size)
            reshaped_summed = summed_top_logprobs.reshape(
                batch_size, self.beam_size * self.per_node_beam_size
            )
            # shape: (batch_size, beam_size * per_node_beam_size)
            reshaped_predicted_classes = predicted_classes.reshape(
                batch_size, self.beam_size * self.per_node_beam_size
            )
            # Append the predictions to the current beam.
            reshaped_beam = (
                predictions.view(batch_size * self.beam_size, 1, -1)
                .repeat(1, self.per_node_beam_size, 1)
                .reshape(batch_size, self.beam_size * self.per_node_beam_size, -1)
            )
            # batch_size, (beam_size * per_node_beach_size), #token
            reshaped_beam = torch.cat([reshaped_beam, reshaped_predicted_classes.unsqueeze(-1)], dim=-1)

            # Keep only the top `beam_size` beam indices.
            # shape: (batch_size, beam_size), (batch_size, beam_size)
            restricted_beam_logprobs, restricted_beam_indices = reshaped_summed.topk(
                self.beam_size
            )
            predictions = reshaped_beam.gather(
                1, restricted_beam_indices.unsqueeze(-1).repeat(1,1,reshaped_beam.shape[-1])
            )

            # shape: (batch_size, beam_size)
            last_logprobs = restricted_beam_logprobs

        if not torch.isfinite(last_logprobs).all():
            warnings.warn(
                "Infinite log probs encountered. Some final captions may not "
                "make sense. This can happen when the beam size is larger than"
                " the number of valid (non-zero probability) transitions that "
                "the step function produces.",
                RuntimeWarning,
            )

        # Optionally select best beam and its logprobs.
        if only_return_best:
            # shape: (batch_size, sequence_length)
            predictions = predictions[:, 0, :]
            last_logprobs = last_logprobs[:, 0]
        num_valid = (predictions != self._eos_index).sum(dim=-1)
        num_valid += (predictions == self._eos_index).sum(dim=-1) > 0
        num_valid = num_valid - start_predictions.shape[1]
        num_valid = num_valid.clip(min=1)

        last_logprobs = last_logprobs / num_valid

        return predictions, last_logprobs
    
    
class GeneratorWithBeamSearch(object):
    def __init__(
        self,
        eos_index: int,
        max_steps: int,
        beam_size: int,
        per_node_beam_size: int = 2,
        length_penalty: float = 1,
        repetition_penalty: float = 1,
        temperature: float = 1,
    ) -> None:
        self._eos_index = eos_index
        self.max_steps = max_steps
        self.beam_size = beam_size
        # NOTE: Expand >1 words to leave some spare tokens to keep the
        # beam size, because some sentences may end here and cannot expand
        # in the next level
        self.per_node_beam_size = per_node_beam_size or beam_size
        self.length_penalty = length_penalty
        self.repetition_penalty = repetition_penalty
        self.temperature = temperature

        assert self.per_node_beam_size > 1
        assert self.length_penalty > 0, "`length_penalty` should be strictely positive."
        assert self.repetition_penalty >= 1.0, "`repetition_penalty` should be >= 1."
        assert self.temperature > 0, "`temperature` should be strictely positive."

    def search(
        self,
        input_ids,
        step,
        num_keep_best= 1,
        do_sample=False,
        top_k=None,
        top_p=None,
        num_return_sequences=1,
    ):
        if num_return_sequences != 1:
            input_ids = input_ids[:, None, :].expand(
                input_ids.shape[0], num_return_sequences, input_ids.shape[1])
            input_ids = input_ids.reshape(-1, input_ids.shape[-1])
        batch_size, cur_len = input_ids.shape
        num_beams = self.beam_size
        pad_token_id = self._eos_index
        eos_token_ids = [self._eos_index]
        per_node_beam_size = self.per_node_beam_size
        repetition_penalty = self.repetition_penalty
        temperature = self.temperature

        # Expand input to num beams
        input_ids = input_ids.unsqueeze(1).expand(batch_size, num_beams, cur_len)
        input_ids = input_ids.contiguous().view(batch_size * num_beams, cur_len)  # (batch_size * num_beams, cur_len)

        #prefix_len = cur_len
        #max_length = self.max_steps + prefix_len
        max_length = self.max_steps
        # generated hypotheses
        generated_hyps = [
            BeamHypotheses(num_keep_best, max_length, self.length_penalty, early_stopping=False) for _ in range(batch_size)
        ]

        # scores for each sentence in the beam
        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view(-1)  # shape (batch_size * num_beams,)

        ## cache compute states
        #past = None

        # done sentences
        done = [False for _ in range(batch_size)]
        while cur_len < max_length:
            scores = step(input_ids)  # (batch_size * num_beams, cur_len, vocab_size)
            vocab_size = scores.shape[-1]

            ## if model has past, then set the past variable to speed up decoding
            #if self._do_output_past(outputs):
                #past = outputs[1]

            # repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858)
            if repetition_penalty != 1.0:
                for i in range(batch_size * num_beams):
                    for previous_token in set(input_ids[i].tolist()):
                        # if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
                        if scores[i, previous_token] < 0:
                            scores[i, previous_token] *= repetition_penalty
                        else:
                            scores[i, previous_token] /= repetition_penalty

            if do_sample:
                # Temperature (higher temperature => more likely to sample low probability tokens)
                if temperature != 1.0:
                    scores = scores / temperature
                # Top-p/top-k filtering
                scores = top_k_top_p_filtering(
                    scores, top_k=top_k, top_p=top_p, min_tokens_to_keep=2
                )  # (batch_size * num_beams, vocab_size)
                # Sample [per_node_beam_size] next words for each beam (so we have some spare tokens and match output of greedy beam search)
                next_words = torch.multinomial(F.softmax(scores, dim=-1),
                        num_samples=per_node_beam_size)  # (batch_size * num_beams, TOPN_PER_BEAM)
                # Compute next scores
                _scores = F.log_softmax(scores, dim=-1)  # (batch_size * num_beams, vocab_size)
                _scores = torch.gather(_scores, -1, next_words)  # (batch_size * num_beams, per_node_beam_size)
                next_scores = _scores + beam_scores[:, None].expand_as(_scores)  # (batch_size * num_beams, per_node_beam_size)
                # Match shape of greedy beam search
                beam_indices = torch.arange(num_beams, device=next_words.device) * vocab_size
                beam_indices = beam_indices.repeat(batch_size, per_node_beam_size)
                next_words = next_words.view(batch_size, per_node_beam_size * num_beams)  # (batch_size, TOPN_PER_BEAM * num_beams)
                next_words = next_words + beam_indices
                next_scores = next_scores.view(batch_size, per_node_beam_size * num_beams)  # (batch_size, TOPN_PER_BEAM * num_beams)
            else:
                # do greedy beam search
                scores = F.log_softmax(scores, dim=-1)  # (batch_size * num_beams, vocab_size)
                assert scores.size() == (batch_size * num_beams, vocab_size)
                # Add the log prob of the new beams to the log prob of the beginning of the sequence (sum of logs == log of the product)
                _scores = scores + beam_scores[:, None].expand_as(scores)  # (batch_size * num_beams, vocab_size)
                # re-organize to group the beam together (we are keeping top hypothesis accross beams)
                _scores = _scores.view(batch_size, num_beams * vocab_size)  # (batch_size, num_beams * vocab_size)
                next_scores, next_words = torch.topk(_scores, per_node_beam_size * num_beams, dim=1, largest=True, sorted=True)

            assert next_scores.size() == next_words.size() == (batch_size, per_node_beam_size * num_beams)

            # next batch beam content
            # list of (batch_size * num_beams) tuple(next hypothesis score, next word, current position in the batch)
            next_batch_beam = []

            # for each sentence
            for batch_ex in range(batch_size):

                # if we are done with this sentence
                done[batch_ex] = done[batch_ex] or generated_hyps[batch_ex].is_done(next_scores[batch_ex].max().item())
                if done[batch_ex]:
                    next_batch_beam.extend([(0, pad_token_id, 0)] * num_beams)  # pad the batch
                    continue

                # next sentence beam content
                next_sent_beam = []
                # next words for this sentence
                for idx, score in zip(next_words[batch_ex], next_scores[batch_ex]):

                    # get beam and word IDs
                    beam_id = idx // vocab_size
                    word_id = idx % vocab_size

                    # end of sentence, or next word
                    if word_id.item() in eos_token_ids or cur_len + 1 == max_length:
                        generated_hyps[batch_ex].add(
                            input_ids[batch_ex * num_beams + beam_id, :cur_len].clone(), score.item()
                        )
                    else:
                        next_sent_beam.append((score, word_id, batch_ex * num_beams + beam_id))

                    # the beam for next step is full
                    if len(next_sent_beam) == num_beams:
                        break

                # update next beam content
                if cur_len + 1 == max_length:
                    assert len(next_sent_beam) == 0
                else:
                    assert len(next_sent_beam) == num_beams

                if len(next_sent_beam) == 0:
                    next_sent_beam = [(0, pad_token_id, 0)] * num_beams  # pad the batch
                next_batch_beam.extend(next_sent_beam)
                assert len(next_batch_beam) == num_beams * (batch_ex + 1)

            # sanity check / prepare next batch
            assert len(next_batch_beam) == batch_size * num_beams
            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
            beam_words = input_ids.new([x[1] for x in next_batch_beam])
            beam_idx = input_ids.new([x[2] for x in next_batch_beam])

            # re-order batch
            input_ids = input_ids[beam_idx, :]
            input_ids = torch.cat([input_ids, beam_words.unsqueeze(1)], dim=-1)

            # update current length
            cur_len = cur_len + 1

            # stop when we are done with each sentence
            if all(done):
                break

        # select the best hypotheses
        tgt_len = torch.ones(batch_size, num_keep_best, dtype=torch.long)
        logprobs = torch.zeros(batch_size, num_keep_best,
                dtype=torch.float).fill_(-1e5).to(input_ids.device)
        all_best = []

        for i, hypotheses in enumerate(generated_hyps):
            best = []
            hyp_scores = torch.tensor([x[0] for x in hypotheses.hyp])
            _, best_indices = torch.topk(hyp_scores,
                    min(num_keep_best, len(hyp_scores)), largest=True)
            for best_idx, hyp_idx in enumerate(best_indices):
                conf, best_hyp = hypotheses.hyp[hyp_idx]
                best.append(best_hyp)
                logprobs[i, best_idx] = conf
                tgt_len[i, best_idx] = len(best_hyp) + 1  # +1 for the <EOS> symbol

            all_best.append(best)

        # generate target batch, pad to the same length
        decoded = input_ids.new(batch_size, num_keep_best, max_length).fill_(pad_token_id)
        for batch_idx, best in enumerate(all_best):
            for best_idx, hypo in enumerate(best):
                decoded[batch_idx, best_idx, : tgt_len[batch_idx, best_idx] - 1] = hypo
                decoded[batch_idx, best_idx, tgt_len[batch_idx, best_idx] - 1] = eos_token_ids[0]
        if num_keep_best == 1:
            decoded = decoded.squeeze(dim=1)
        return decoded, logprobs

class BeamHypotheses(object):
    def __init__(self, n_hyp, max_length, length_penalty, early_stopping):
        """
        Initialize n-best list of hypotheses.
        """
        self.max_length = max_length - 1  # ignoring bos_token
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.n_hyp = n_hyp
        self.hyp = []
        self.worst_score = 1e9

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.hyp)

    def _length_norm(self, length):
        #return length ** self.length_penalty
        # beam search alpha: https://opennmt.net/OpenNMT/translation/beam_search/
        return (5 + length) ** self.length_penalty / (5 + 1) ** self.length_penalty

    def add(self, hyp, sum_logprobs):
        """
        Add a new hypothesis to the list.
        """
        #score = sum_logprobs / len(hyp) ** self.length_penalty
        score = sum_logprobs / self._length_norm(len(hyp))
        if len(self) < self.n_hyp or score > self.worst_score:
            self.hyp.append((score, hyp))
            if len(self) > self.n_hyp:
                sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.hyp)])
                del self.hyp[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs):
        """
        If there are enough hypotheses and that none of the hypotheses being generated
        can become better than the worst one in the heap, then we are done with this sentence.
        """
        if len(self) < self.n_hyp:
            return False
        elif self.early_stopping:
            return True
        else:
            #return self.worst_score >= best_sum_logprobs / self.max_length ** self.length_penalty
            return self.worst_score >= best_sum_logprobs / self._length_norm(self.max_length)