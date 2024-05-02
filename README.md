# WS TCG Card Text Translator
A Japanese-English machine translation model specifically trained for translating card text from the Weiss Schwarz (WS) Trading Card Game, fine-tuned on [Helsinki-NLP/opus-mt-ja-en](https://huggingface.co/Helsinki-NLP/opus-mt-ja-en).

## Hugging Face Space Demo
Check out the demo at [https://huggingface.co/spaces/eepj/wstcg-mt](https://huggingface.co/spaces/eepj/wstcg-mt).


## Dataset
### Official WS Card List
* Japanese-English parallel card text comprising 6000+ card text retrieved from the offical card list.

## Training
### Base Model
* Helsinki-NLP/opus-mt-ja-en

### Base Tokenizer
* Helsinki-NLP/opus-mt-ja-en

### Additional Tokens
|Token Type|Additional Tokens|
|----------|-----------------|
|Named Entity Placeholder|\<TRAIT\>, \<NAME\>|
|Trigger Icon Placeholder|\<SOUL\>, \<CHOICE\>, \<TREASURE\>, \<SALVAGE\>, \<STANDBY\>,<br> \<GATE\>, \<BOUNCE\>, \<STOCK\>, \<SHOT\>, \<DRAW\>|
|Keywords|【, 】, AUTO, ACT, CONT, COUNTER, CLOCK, トリガー

