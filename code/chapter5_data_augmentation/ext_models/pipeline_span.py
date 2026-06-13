from .argument_span import ArgumentSpanModel
from .trigger_span import TriggerDetector


def _argument_probes(samples, trigger_predictions):
    probes = []
    index = []
    for s in samples:
        for i, (start, end, full) in enumerate(trigger_predictions.get(s["sen_id"], [])):
            probes.append(
                {
                    "sen_id": s["sen_id"],
                    "event_id": i,
                    "text": s["text"],
                    "trigger_span": (start, end),
                    "event_type": full,
                    "arguments": [],
                }
            )
            index.append((s["sen_id"], i, start, end, full))
    return probes, index


class PipelineSpan:
    def __init__(self, encoder_name, type_system, trigger_cfg, argument_cfg):
        self.type_system = type_system
        self.trigger = TriggerDetector(encoder_name, type_system, trigger_cfg)
        self.argument = ArgumentSpanModel(encoder_name, type_system, argument_cfg)

    def fit(self, train_samples, train_arg_examples):
        self.trigger.fit(train_samples)
        self.argument.fit(train_arg_examples)
        return self

    def predict(self, samples):
        trigger_predictions = self.trigger.predict(samples)
        probes, index = _argument_probes(samples, trigger_predictions)
        arg_pred = self.argument.predict(probes) if probes else {}
        per_sentence = {s["sen_id"]: [] for s in samples}
        for sen_id, eid, start, end, full in index:
            args = arg_pred.get((sen_id, eid), set())
            per_sentence[sen_id].append(
                {
                    "sen_id": sen_id,
                    "trigger": (start, end),
                    "type": full,
                    "arguments": frozenset(args),
                }
            )
        return [per_sentence[s["sen_id"]] for s in samples]
