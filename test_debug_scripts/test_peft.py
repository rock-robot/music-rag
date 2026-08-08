import transformers, peft
print("transformers:", transformers.__version__)   # MUST still say 4.29.2
print("peft        :", peft.__version__)
from anticipation.convert import midi_to_events     # AMT stack still intact?
print("anticipation import OK")