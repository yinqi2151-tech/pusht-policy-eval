"""Run LeRobot 0.6.1's migrate_policy_normalization on legacy checkpoints.

LeRobot's migration script passes config.json values straight into the policy
config constructor. JSON has no tuples, so a legacy ``"crop_shape": [84, 84]``
stays a list, and ``draccus`` then fails to encode it as ``tuple[int, int]``
when saving (``Exception: Couldn't encode 84``). This wrapper converts lists to
tuples only for fields annotated as tuples, then runs the unmodified migration.
All arguments are forwarded to the original script. A failure to build the
optional model card (it needs the Hub) is reported but does not abort.
"""

import dataclasses
import typing

from lerobot.processor import migrate_policy_normalization as migrate
from lerobot.configs.policies import PreTrainedConfig

_original_make_policy_config = migrate.make_policy_config


def _is_tuple_type(tp) -> bool:
    if tp is tuple or typing.get_origin(tp) is tuple:
        return True
    return any(_is_tuple_type(arg) for arg in typing.get_args(tp))


def make_policy_config_with_tuples(policy_type: str, **kwargs):
    config_cls = PreTrainedConfig.get_choice_class(policy_type)
    hints = typing.get_type_hints(config_cls)
    converted = []
    for field in dataclasses.fields(config_cls):
        value = kwargs.get(field.name)
        if isinstance(value, list) and _is_tuple_type(hints.get(field.name)):
            kwargs[field.name] = tuple(value)
            converted.append(field.name)
    if converted:
        print(f"Converted JSON lists to tuples for: {converted}")
    return _original_make_policy_config(policy_type, **kwargs)


class _SkippedCard:
    def save(self, path):
        print(f"Model card skipped (could not be generated); weights and processors are already saved.")


def _wrap_model_card(original):
    def generate_model_card(self, *args, **kwargs):
        # Card generation calls the Hub's YAML validator; a network failure here
        # must not discard a migration whose files were already written.
        try:
            return original(self, *args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING: model card generation failed: {exc!r}")
            return _SkippedCard()

    return generate_model_card


if __name__ == "__main__":
    from lerobot.policies.pretrained import PreTrainedPolicy

    migrate.make_policy_config = make_policy_config_with_tuples
    PreTrainedPolicy.generate_model_card = _wrap_model_card(PreTrainedPolicy.generate_model_card)
    migrate.main()
