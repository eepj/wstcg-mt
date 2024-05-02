# WS TCG Card Text Translator
A Japanese-English machine translation model specifically trained for translating card text from the Weiss Schwarz (WS) Trading Card Game, fine-tuned on [Helsinki-NLP/opus-mt-ja-en](https://huggingface.co/Helsinki-NLP/opus-mt-ja-en).

## Hugging Face Hub Model
Check out the model at [https://huggingface.co/eepj/wstcg-mt-ja-en](https://huggingface.co/eepj/wstcg-mt-ja-en).

## Dataset
### Official WS Card List
* Japanese-English parallel card text comprising 6000+ card text retrieved from the offical card list.

## Training
### Base Model
* Base model: Helsinki-NLP/opus-mt-ja-en
* Base tokenizer: Helsinki-NLP/opus-mt-ja-en
* Source language: Japanese (ja)
* Target language: English (en)

### Additional Tokens
|Token Type|Additional Tokens|
|----------|-----------------|
|Named Entity Placeholder|\<TRAIT\>, \<NAME\>|
|Trigger Icon Placeholder|\<SOUL\>, \<CHOICE\>, \<TREASURE\>, \<SALVAGE\>, \<STANDBY\>,<br> \<GATE\>, \<BOUNCE\>, \<STOCK\>, \<SHOT\>, \<DRAW\>|
|Keywords|【, 】, AUTO, ACT, CONT, COUNTER, CLOCK, トリガー|

### Hardware
* NVIDIA RTX3060 Ti with CUDA hardware acceleration

### Hyperparameters
* Number of epochs: 5
* Optimizer: Adam
* Initial learning rate: 1e-4
* Learning rate scheduler: StepLR, reduce by factor of 0.5 every epoch
* Batch size: 4
* Loss function: CrossEntropyLoss
* Random seed: 42

## Performance
### Metrics
|Dataset|BLEU|chr-F|
|-------|------|-----|
|WS Official Card List|0.82664|0.96515|

### Example Test Case
|Language|Official Text|
|--------|-------------|
|Japanese|【永】 あなたの\<TRAIT\>のキャラが4枚以上なら、このカードは、色条件を満たさずに手札からプレイでき、あなたの手札のこのカードのレベルを－1。|
|English|【CONT】 If you have 4 or more \<TRAIT\> characters, this card gets -1 level while in your hand, and can be played from your hand without fulfilling color requirements.|

|Model|Translated Text|BLEU|chr-F|
|-----|---------------|------|-----|
|Ours|【CONT】If you have 4 or more \<TRAIT\> characters, this card can be played from your hand without fulfilling color requirements, and this card gets -1 level while in your hand.|0.79134|0.95207|
|Google Translate|[CONT] If you have 4 or more characters in \<TRAIT\>, you can play this card from your hand without meeting the color conditions, and the level of this card in your hand becomes -1.|0.26561|0.53877|
|GPT-3.5|[Permanent] If you have 4 or more cards with the \<TRAIT\> trait, you can play this card from your hand without meeting the color condition, and reduce the level of this card in your hand by 1.|0.24213|0.48925|
|opus-mt-ja-en|If there are more than four characters in your \<TRAIT\> card, this card can be played from your hand with no color conditions, and you can see the level of this card in your hand is -1.|0.22006|0.51034|

## References
**Helsinki-NLP/opus-mt-ja-en**
<br>
https://huggingface.co/Helsinki-NLP/opus-mt-ja-en

**WS Official Card List (Japanese)**
<br>
https://ws-tcg.com/cardlist

**WS Official Card List (English)**
<br>
https://en.ws-tcg.com/cardlist

**WS Comprehensive Rules (English)**
<br>
https://en.ws-tcg.com/wp/wp-content/uploads/20220726205726/WSE-Comprehensive-Rules-v2.07.pdf