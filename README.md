# Advanced Voice AI Stack: Interview Questions & Answers

## إزاي تستخدم الملف ده

الأسئلة دي معمولة لمقابلات senior/staff/lead level على الـ voice AI stack بالكامل: audio fundamentals، STT، TTS، real-time voice agents، speech LLMs، serving، data، security، debugging، تدريب foundation models من الصفر، الأسئلة النظرية، والأسئلة القيادية. كل سؤال معاه إجابة نموذجية بتغطي النقط اللي المفروض المرشح القوي يلمسها، وسؤال متابعة تستخدمه لو عايز تعرف عمق فهمه. مش مطلوب إن المرشح يقول كل حاجة في الإجابة، المهم إنه يشرح الـ tradeoffs بوضوح، ويربط الكلام بحاجة عملها فعلًا، ويعرف يقول "مش عارف" لما ميعرفش.

الملف فيه 168 سؤال مفصّل في 12 قسم، زائد 45 سؤال سريع للـ screening و7 تمارين عملية. في مقابلة ساعة، اختار 6 إلى 8 أسئلة من أقسام مختلفة، وخلي واحد منهم على الأقل system design. الاختيار حسب الدور:

- مرشح foundation models (تدريب من الصفر): ركّز على الأقسام 2 و3 و5 و7 و10 و11.
- مرشح production stack (agents وserving): ركّز على الأقسام 4 و6 و8 و9.
- مرشح lead: ضيف سؤالين من القسم 12 وسؤال debugging من القسم 9.
- للـ screening على التليفون: 15 سؤال عشوائي من القسم 13.
- للجزء العملي: تمرين واحد من القسم 14 live، أو اتنين كـ take-home.

الأسئلة مبنية على السوق العربي والخليجي بالتحديد: اللهجات، التشكيل، الـ normalization، التليفون 8 kHz، الـ on-prem، الـ licenses، والامتثال (PDPL وSAMA). الأرقام المذكورة في الإجابات (latency، أحجام الـ data، عدد الـ streams) أرقام إرشادية من الممارسة مش حدود ثابتة، والمرشح اللي بيقول "بقيسها على الـ setup بتاعي" بيجاوب صح.

## المحتويات

- 1. أساسيات الصوت والـ signal processing (س1 إلى س11)
- 2. الـ Speech-to-Text (ASR) (س12 إلى س37)
- 3. الـ Text-to-Speech (TTS) (س38 إلى س58)
- 4. الـ real-time voice agents (س59 إلى س83)
- 5. الـ speech LLMs (س84 إلى س92)
- 6. الـ serving والـ infrastructure (س93 إلى س104)
- 7. الـ data (س105 إلى س116)
- 8. الأمان والخصوصية والـ safety (س117 إلى س123)
- 9. الـ debugging والخبرة العملية (س124 إلى س133)
- 10. تدريب foundation speech models من الصفر (س134 إلى س145)
- 11. أسئلة نظرية وبحثية (س146 إلى س154)
- 12. أسئلة قيادية وbusiness لمستوى lead (س155 إلى س161)
- 13. أسئلة سريعة للـ screening (45 سؤال سريع)
- 14. تمارين عملية وcoding (س162 إلى س168)
- ملاحظات سريعة للتقييم

---

## 1. أساسيات الصوت والـ signal processing

### س1. ايه الفرق بين الـ waveform والـ spectrogram والـ mel spectrogram؟ وليه أغلب موديلات الكلام بتشتغل على mel مش على الـ raw waveform؟

**الإجابة النموذجية:**
الـ waveform هو الـ amplitude بتاع الصوت مع الزمن بعد الـ sampling (مثلًا 16,000 sample في الثانية عند 16 kHz) والـ quantization (16-bit PCM غالبًا). الـ spectrogram بيطلع من الـ STFT: بتقطع الإشارة لـ windows قصيرة (مثلًا 25 ms) متداخلة بـ hop (مثلًا 10 ms)، وتعمل FFT لكل window، فبتحصل على magnitude لكل frequency bin في كل frame. الـ mel spectrogram بيمرر الـ power spectrum على mel filterbank (80 أو 128 filter) بتضغط الترددات العالية وتفصّل الترددات الواطية زي ما ودن الإنسان بتسمع، وبعدين log compression عشان الـ dynamic range.

ليه mel؟ لأنه بيقلل الـ dimensionality جدًا (من 400 sample للـ window لـ 80 قيمة)، وبيركز على المعلومات المهمة للـ perception، وبيخلي الـ training أسهل وأسرع. الـ raw waveform models موجودة (الـ CNN feature encoder في wav2vec 2.0 بيشتغل على raw audio مباشرة)، بس لسه الـ mel هو الـ default في Whisper وأغلب TTS.

حاجة مهمة المرشح المفروض يذكرها: الـ frame rate. عند 16 kHz وhop 160 sample، كل frame بيمثل 10 ms، يعني 100 frame في الثانية، وده اللي بيحدد طول الـ sequence اللي بيدخل الـ encoder، وبيتضغط بعد كده بالـ subsampling (Whisper بيوصل لـ 50 frame/s بعد الـ conv stem، والـ Conformer بيعمل 4x subsampling يعني 40 ms لكل frame، والـ FastConformer بيعمل 8x). كمان الـ mel بيرمي الـ phase، وعشان كده الـ vocoder في TTS شغلته إنه يعيد بناء الـ phase من الـ magnitude.

**سؤال متابعة:** ليه بنستخدم log في الـ mel؟ وايه اللي يحصل لو غيرت الـ hop length في موديل متدرب على hop معين؟

### س2. الفرق بين 8 kHz و16 kHz و24 kHz و44.1/48 kHz، وايه تأثير ده على STT وTTS؟

**الإجابة النموذجية:**
الـ Nyquist بيقول إن أعلى تردد تقدر تمثله هو نص الـ sample rate. التليفون التقليدي بيشتغل على 8 kHz (narrowband) يعني الصوت مقصوص عند 3.4 kHz تقريبًا، وده بيضيّع معلومات مهمة للـ fricatives زي س وث وف وش، وبالنسبة للعربي الفرق بين س وث بالذات بيتأذي. أغلب موديلات الـ STT متدربة على 16 kHz (wideband)، وده اللي بيغطي الكلام كويس. الـ TTS بيطلع 22.05 أو 24 kHz غالبًا، والـ studio quality 44.1/48 kHz.

النقطة اللي بتفرق في المقابلة: لو عندك audio من telephony بـ 8 kHz وعملت upsampling لـ 16 kHz، انت مضفتش أي معلومة، الموديل هيشوف mel spectrogram نصه الأعلى فاضي، وده distribution shift بيرفع الـ WER. الحل: fine-tuning أو augmentation بمحاكاة الـ telephony channel (downsample لـ 8k، G.711 codec، بعدين upsample) وقت التدريب، أو موديل مخصص للـ narrowband. وفي التقييم لازم يكون عندك test set من نفس الـ channel.

في الـ TTS العكس: لو الموديل بيطلع 24 kHz وهيتبعت على تليفون، هيتعمله downsample وهيخسر الـ brightness، فمفيش داعي تدفع compute لـ 48 kHz vocoder في use case تليفوني.

**سؤال متابعة:** لو عندك call recordings مسجلة stereo (قناة لكل متكلم)، ده بيغير ايه في الـ pipeline؟ (الإجابة المتوقعة: بتلغي الحاجة للـ diarization تقريبًا).

### س3. ايه الفرق بين الـ VAD والـ endpointing؟ واشرح الفرق بين energy-based VAD وWebRTC VAD وSilero VAD.

**الإجابة النموذجية:**
الـ VAD (Voice Activity Detection) بيجاوب على سؤال واحد لكل frame قصير (10 إلى 30 ms): في كلام ولا لأ. الـ endpointing سؤال أعلى مستوى: المستخدم خلّص كلامه ولا لسه؟ الـ VAD مدخل للـ endpointing، بس الـ endpointing محتاج logic زيادة: min silence duration بعد آخر speech، hangover، وفي الأنظمة الحديثة semantic signals من الـ transcript نفسه.

الـ energy-based VAD بيقارن الـ RMS energy بـ threshold، رخيص جدًا بس بيفشل مع الـ noise والـ music. الـ WebRTC VAD مبني على GMM على features بسيطة، سريع وخفيف، بس بيتخدع بالضوضاء العالية. الـ Silero VAD شبكة عصبية صغيرة جدًا (أقل من 2 MB) بتشتغل على chunks حوالي 30 ms وبتدي احتمال، وأدق بكتير في الضوضاء، وبقت الـ default في أغلب الـ voice agent frameworks. فيه كمان الـ VAD اللي جوه الـ STT نفسه (Whisper بيطلع no_speech probability بس ده مش مناسب للـ real-time).

النقطة الأهم إنه يفهم إن الـ VAD threshold والـ min silence بيتحكموا في latency الـ agent بشكل مباشر، والموضوع ده هنرجعله في قسم الـ agents.

**سؤال متابعة:** ليه الـ VAD مهم قبل Whisper بالذات؟ (عشان الـ hallucination على الصمت، وتوفير compute).

### س4. AEC وnoise suppression وAGC: ايه كل واحدة، وترتيبهم في الـ pipeline، وامتى الـ noise suppression بيضر؟

**الإجابة النموذجية:**
الـ AEC (Acoustic Echo Cancellation) بيشيل صوت الـ agent نفسه لما يرجع من السماعة للمايك، وده شرط أساسي لأي barge-in، لأنه من غيره الـ VAD هيعتبر صوت الـ TTS كلام من المستخدم. بيشتغل بـ adaptive filter (NLMS) على الـ reference signal (اللي انت بتشغله) وبيطرحه من الـ mic signal، وفيه نسخ neural حديثة. الـ NS (Noise Suppression) بيشيل الضوضاء الثابتة والمتغيرة، من RNNoise لـ DeepFilterNet وموديلات DNS challenge. الـ AGC بيوحّد مستوى الصوت.

الترتيب المعتاد: AEC أولًا (لأنه محتاج الإشارة الأصلية عشان يعمل الـ alignment) ثم NS ثم AGC. في المتصفح الـ WebRTC بيعمل الثلاثة تلقائيًا في الـ client، وده سبب كبير لتفضيل WebRTC على WebSocket في الـ browser.

امتى الـ NS بيضر؟ لما يكون aggressive فيشوّه الكلام ويشيل consonants خفيفة، فالـ STT اللي متدرب على clean speech بيسوء. القاعدة العملية: خلّي الـ STT نفسه robust للضوضاء بالـ augmentation بدل ما تعتمد على NS قوي قبله، واستخدم NS خفيف أو اطفيه في الـ server-side لو الـ STT شغال كويس. وفي التدريب، لو هتشغل NS في الـ production لازم تشغله على training data كمان عشان الـ distribution تتطابق.

**سؤال متابعة:** في telephony مفيش client-side AEC، بتعمل ايه؟ (server-side AEC على الـ media server، أو echo cancellation على مستوى الـ PBX، أو تعتمد على half-duplex logic).

### س5. الـ STFT parameters: ليه 25 ms window و10 ms hop بالذات؟ واشرح الـ window functions والـ spectral leakage والـ time-frequency tradeoff.

**الإجابة النموذجية:**
الـ window length بيحدد الـ frequency resolution، والـ hop بيحدد الـ time resolution. عند 16 kHz، الـ window بتاع 25 ms هو 400 sample، ولما تعمل عليه FFT بحجم 512 بتحصل على 257 frequency bin بمسافة حوالي 31 Hz بين كل bin. لو كبّرت الـ window لـ 100 ms هتحسّن الـ frequency resolution بس هتضيّع التفاصيل الزمنية السريعة زي الـ plosives (ب، ت، ك، ق)، ولو صغّرتها لـ 5 ms هتشوف الوقت كويس بس الـ harmonics هتتلخبط. الـ 25 ms اختيار تاريخي مبني على إن الكلام بيعتبر quasi-stationary في المدى ده، والـ 10 ms hop بيدي overlap 60% وبيخلي الـ frames تتبع تغير الـ formants بسلاسة.

الـ window function (Hann أو Hamming غالبًا، مش rectangular) بتقلل الـ spectral leakage: لو قطعت الإشارة بحدود حادة، الـ FFT بيفترض إنها بتتكرر periodic وبيطلع energy وهمي متوزع على كل الترددات. الـ Hann بيخفف الأطراف لصفر فبيقلل الـ leakage على حساب شوية main lobe width. في الـ TTS الأرقام بتختلف حسب الـ sample rate: موديلات كتير عند 24 kHz بتستخدم n_fft 1024 وhop 256 (حوالي 10.7 ms، يعني قرابة 94 frame في الثانية) و100 mel bin، والـ vocoders اللي بتشتغل عند 44.1 kHz زي BigVGAN بتستخدم hop 512 و128 mel.

اللي بيفرق في المقابلة: إن المرشح يفهم إن الـ mel parameters جزء من عقد الموديل. الـ vocoder متدرب على mel بمواصفات معينة (sample rate، n_fft، hop، عدد الـ bins، fmin/fmax، نوع الـ log والـ normalization)، ولو الـ acoustic model طلّع mel بمواصفات مختلفة ولو بفرق بسيط، الصوت هيطلع مشوّه أو متكسر بدون error واضح. وفي الـ STT نفس الكلام: الـ feature extractor لازم يكون هو هو في التدريب والـ inference بما فيه طريقة الـ padding والـ center=True.

**سؤال متابعة:** لو عندك موديل STT متدرب على 80 mel وعايز تجرب 128 mel، ايه اللي لازم يتغير في الشبكة وايه اللي ممكن يتحسن؟

### س6. MFCC مقابل log-mel filterbank مقابل الـ learned frontends: ليه الـ MFCC اختفى من الموديلات الحديثة؟ وايه دور الـ CMVN؟

**الإجابة النموذجية:**
الـ MFCC هو log-mel وبعده DCT بياخد أول 13 أو 20 coefficient. الـ DCT كان مهم في عصر الـ GMM-HMM لأنه بيعمل decorrelation للـ features فبيسمح باستخدام diagonal covariance، وبيضغط الـ dimensionality. الشبكات العصبية مش محتاجة الـ decorrelation ده، بالعكس، الـ CNN والـ attention بيستفيدوا من الـ local correlations بين الـ mel bins المتجاورة، والـ DCT بيرمي معلومات. عشان كده أغلب الموديلات الحديثة بتاخد log-mel مباشرة (80 أو 128 bin).

الـ learned frontends زي SincNet أو الـ conv feature encoder في wav2vec 2.0 بتتعلم الـ filters من الـ raw waveform. النظرية إن الشبكة تتعلم filterbank أحسن من الـ mel، وعمليًا الفرق مش كبير في أغلب الـ benchmarks، بس بتفيد في حالات معينة زي الـ SSL pretraining لأن الموديل مش مقيّد بالـ mel prior. التكلفة إن الـ raw waveform أطول 320 مرة من الـ mel frames، فالـ conv stem بياخد compute معتبر.

الـ CMVN (cepstral mean and variance normalization) بيطرح الـ mean ويقسم على الـ std لكل feature dimension، إما global (إحصائيات من الـ training set كله) أو per-utterance. الـ per-utterance بيلغي تأثير الـ channel والـ recording gain بشكل أقوى، لكنه مستحيل في الـ streaming لأنك مش شايف الـ utterance كلها، فبتستخدم global normalization أو running statistics causal. Whisper بيستخدم normalization ثابتة: log-mel مقصوص عند max minus 8 ومقسوم على 4 بعد إزاحة، ودي كمان لازم تتكرر بالظبط في الـ inference.

**سؤال متابعة:** موديل متدرب بـ per-utterance CMVN وعايز تشغله streaming. ايه اللي هيحصل لو بدلت لـ global normalization من غير retraining؟ (الإجابة المتوقعة: degradation ملحوظ، والحل إنك تعيد التدريب أو على الأقل fine-tuning بالـ normalization الجديدة).

### س7. اشرح الـ source-filter model والـ F0 والـ formants والـ harmonics. ايه اللي بيحمل هوية المتكلم وايه اللي بيحمل المحتوى اللغوي؟ وايه خصوصية الأصوات العربية زي الحروف المفخمة والحلقية؟

**الإجابة النموذجية:**
الـ source-filter model بيقول إن الكلام هو مصدر (اهتزاز الأحبال الصوتية بتردد أساسي F0 للأصوات المجهورة، أو ضوضاء للمهموسة) بيتفلتر بالـ vocal tract (الحلق والفم والأنف). الـ vocal tract بيعمل resonances اسمها formants (F1، F2، F3)، وموضعها هو اللي بيحدد الـ vowel: الفتحة والكسرة والضمة بيتفرقوا أساسًا بـ F1 وF2. الـ harmonics هي مضاعفات الـ F0 وبتظهر كخطوط أفقية في الـ spectrogram، والـ formants هي المناطق اللي فيها الـ harmonics أقوى.

المحتوى اللغوي بيعيش أساسًا في الـ formant trajectories والـ spectral envelope السريعة التغير. هوية المتكلم بتعيش في متوسط الـ F0 ومداه، وطول الـ vocal tract (اللي بيزيح الـ formants كلها)، والـ spectral tilt، وخصائص الـ glottal source، وطبعًا العادات النطقية. الفصل ده هو أساس الـ voice conversion والـ speaker embeddings: الـ ECAPA بيحاول يلخص خصائص المتكلم بغض النظر عن الكلام، والـ content encoders (زي HuBERT layers الوسطى) بتحاول العكس.

في العربي فيه مجموعتين مهمين: الحروف المفخمة (ص ض ط ظ) بتتميز عن نظائرها (س د ت ذ) أساسًا بانخفاض F2 في الـ vowel المجاورة بسبب الـ pharyngealization، والحروف الحلقية (ع ح) بتتميز بتضييق في الحلق بيبان في F1 مرتفع وتغير في الـ spectral shape. الفرق بين ق وك وبين ط وت بيعيش في الـ vowel المجاورة أكتر ما بيعيش في الحرف نفسه. ده بيفسر ليه الموديلات المتدربة على لغات أوروبية بتحتاج data عربي كتير عشان تلقط الفروق دي، وليه في اللهجات (نطق ق كـ g في نجد، أو كـ hamza في مصر والحجاز الحضري) الموديل محتاج يشوف الـ variation ده صراحة.

**سؤال متابعة:** ليه الـ pitch shifting البسيط بيبوظ طبيعية الصوت لو ما غيرتش الـ formants معاه؟ (لأنك بتزيح الـ source والـ filter مع بعض، فبيطلع صوت "chipmunk" لأن الـ vocal tract length بقى غير واقعي).

### س8. الـ audio codecs في التليفون والإنترنت: G.711 وG.722 وG.729 وAMR وOpus وAAC. ازاي الـ lossy compression بيأثر على الـ STT، وازاي تعمل codec augmentation؟

**الإجابة النموذجية:**
الـ G.711 هو codec التليفون الكلاسيكي: 8 kHz، 8-bit companding (μ-law في أمريكا وA-law في أغلب العالم بما فيه السعودية)، 64 kbps، مفيش compression حقيقي غير الـ companding فالتشويه محدود بس الـ bandwidth مقصوص. الـ G.722 wideband عند 16 kHz وبيطلع في HD voice. الـ G.729 وAMR-NB عندهم compression قوي (8 kbps وأقل) وبيدخلوا artifacts واضحة، والـ AMR-WB (HD voice في الموبايل) أحسن. الـ Opus هو الـ default في WebRTC: بيغطي من 6 لـ 510 kbps، وبيجمع SILK للكلام وCELT للموسيقى، وعنده FEC وDTX وPLC مدمجين. الـ AAC والـ MP3 بتقابلهم في الـ found data (YouTube وpodcasts)، وبيقصوا الترددات العالية عند الـ bitrates الواطية.

التأثير على الـ STT: الـ band-limiting بيضيّع الـ fricatives، والـ compression artifacts بتظهر كـ patterns مش موجودة في الـ clean speech، والـ packet loss concealment بيعمل تكرارات وامتدادات غريبة. لو الموديل ما شافش ده في التدريب، الـ WER بيقفز على المكالمات الحقيقية حتى لو كان ممتاز على الـ test sets النظيفة.

الـ codec augmentation: وقت التدريب، لنسبة من الـ batches، تمرر الصوت على chain بتحاكي القناة: resample لـ 8k، G.711 encode/decode (أو AMR أو Opus عند bitrate واطي)، رجّعه لـ 16k. بتتعمل بـ ffmpeg أو sox أو مكتبات زي audiomentations وtorchaudio (اللي عندها codec functions). المهم إنك تحاكي الـ chain كاملة مش خطوة واحدة، وتضيف packet loss محاكى لو الـ use case فيه VoIP. والمقياس النهائي دايمًا test set من نفس القناة الحقيقية مش المحاكاة.

**سؤال متابعة:** لو الـ TTS بتاعك هيتبعت على تليفون G.711، ايه اللي تعمله في الـ post-processing قبل الإرسال؟ (band-limit ثم downsample بشكل صحيح، وتعديل الـ loudness عشان الـ companding، والتأكد من عدم الـ clipping).

### س9. الـ loudness والـ normalization والـ resampling: dBFS وRMS وLUFS، والـ clipping والـ DC offset، وليه الـ resampling library ممكن يبوظ الموديل؟

**الإجابة النموذجية:**
الـ dBFS بيقيس الـ peak بالنسبة لأقصى قيمة رقمية (0 dBFS)، والـ RMS بيقيس الطاقة المتوسطة، والـ LUFS (من EBU R128 وITU-R BS.1770) بيقيس الـ perceived loudness مع فلتر بيحاكي الأذن وgating للصمت. الـ peak normalization بتضمن مفيش clipping بس مش بتوحّد الإحساس بالعلو، والـ loudness normalization (مثلًا لـ -23 LUFS في البث أو -16 في الـ streaming) بتوحّده. للـ TTS datasets بتعمل loudness normalization عشان الموديل ما يتعلمش تغيرات الـ gain كأنها style. للـ STT الموديلات الحديثة robust للـ gain نسبيًا بس الـ AGC augmentation بتفيد.

الـ clipping (الإشارة قصت عند الـ max) بيدخل harmonics حادة وبيكسر الـ STT والـ TTS، ولازم تكتشفه في الـ QA (نسبة الـ samples اللي عند الـ full scale). الـ DC offset (متوسط الإشارة مش صفر) بيظهر كـ energy عند 0 Hz وبيبوظ بعض الـ vocoders، والحل high-pass filter بسيط.

الـ resampling: تحويل 44.1 لـ 16 kHz لازم يعمل low-pass anti-aliasing قبل الـ decimation. المكتبات المختلفة (sox، libsamplerate، soxr، librosa، torchaudio مع lowpass_filter_width مختلف، scipy) بتطلع نتائج مش متطابقة، وفيه implementations بتعمل aliasing خفيف. لو الـ training pipeline استخدم مكتبة والـ serving استخدم مكتبة تانية، بتحصل على WER أعلى بشكل يصعب تفسيره. القاعدة: مكتبة واحدة مثبتة الإصدار في الاتنين، وتحفظ الـ configuration مع الموديل. نفس الحاجة في تحويل int16 لـ float (القسمة على 32768) وفي دمج الـ stereo لـ mono (متوسط القناتين ولا قناة واحدة؟).

**سؤال متابعة:** ازاي تكتشف إن ملف 16 kHz أصله كان 8 kHz واتعمله upsampling؟ (spectral rolloff: مفيش energy تقريبًا فوق 4 kHz).

### س10. الـ far-field والـ reverberation والـ multi-microphone: RT60 وRIR augmentation والـ dereverberation والـ beamforming. امتى تحتاجهم؟

**الإجابة النموذجية:**
لما المايك بعيد عن المتكلم (قاعة اجتماعات، سماعة ذكية، سيارة)، الصوت المباشر بيوصل ومعاه انعكاسات من الحيطان والسقف. الـ RT60 هو الوقت اللي الطاقة بتنقص فيه 60 dB، وبيتراوح من 0.2 ثانية في غرفة مفروشة لأكتر من ثانية في قاعة فاضية. الـ reverberation بيمرّ الـ phonemes على بعضها (temporal smearing) وبيقلل الـ direct-to-reverberant ratio، فالـ STT المتدرب على close-talk بيتدهور بشدة.

الحل الأول data-centric: RIR augmentation، يعني convolution للصوت النظيف مع room impulse responses حقيقية (مجموعات زي الـ RIRs المستخدمة في Kaldi) أو محاكاة (pyroomacoustics بالـ image method). المرشح القوي بيذكر إنك لازم تنوّع الـ RT60 وموضع المصدر والمايك، وإن الـ RIR الحقيقي أحسن من المحاكاة لأن المحاكاة بتفتقد التفاصيل، وإنك ممكن تفصل early reflections (أول 50 ms، بتساعد الفهم) عن الـ late reverberation (اللي بتضر).

الـ dereverberation بـ WPE (weighted prediction error) شغّال كويس كـ preprocessing خصوصًا مع multi-channel. الـ beamforming بيستخدم أكتر من مايك: delay-and-sum بسيط، والـ MVDR بيقمع الضوضاء من اتجاهات معينة، والـ mask-based beamforming بيستخدم شبكة عصبية عشان تقدّر الـ masks اللي بتتحسب منها الـ covariance matrices. بتحتاجهم في الاجتماعات والأجهزة، مش في التليفون (قناة واحدة) ولا في الـ headset. وفي الـ TTS data collection، الـ reverberation عدو: التسجيل لازم يكون في بيئة جافة (treated room) لأن الموديل هيتعلم الغرفة كجزء من الصوت.

**سؤال متابعة:** عندك تسجيلات اجتماعات من مايك واحد في نص الترابيزة، والـ WER عالي. ايه ترتيب الحاجات اللي تجربها؟

### س11. ازاي تقيّم جودة الـ audio في dataset قبل التدريب؟ SNR estimation وDNSMOS والـ bandwidth detection والـ clipping والـ silence ratio.

**الإجابة النموذجية:**
الجودة بتتقاس على محاور منفصلة: (1) الـ SNR: من غير مرجع نظيف بتقدّره بطرق زي WADA-SNR أو بحساب energy الكلام مقابل energy الصمت بالـ VAD. (2) الجودة الإدراكية: DNSMOS (P.835) بيدي SIG للكلام وBAK للخلفية وOVRL إجمالي، وUTMOS وNISQA للـ TTS data. (3) الـ bandwidth: spectral rolloff بيكشف الملفات اللي أصلها 8 kHz أو الـ MP3 المقصوص. (4) الـ clipping: نسبة الـ samples عند الـ full scale. (5) الـ silence ratio والـ leading/trailing silence. (6) الـ duration distribution: الـ segments القصيرة جدًا أو الطويلة جدًا بتعمل مشاكل في الـ batching. (7) الـ speech/music/noise classification لفلترة الـ found data. (8) عدد المتكلمين في الـ segment (overlap detection) لأن الـ TTS data لازم يكون single-speaker.

الـ thresholds بتختلف حسب الهدف: الـ TTS محتاج data شديد النظافة (DNSMOS عالي، SNR فوق 30-35 dB، مفيش reverberation)، والـ STT بالعكس محتاج تنوع فيه noise لأن الـ production فيه noise، فهنا الفلترة بتكون بس لإزالة الحالات المكسورة (ملفات فاضية، clipping شديد، transcripts مش متطابقة).

النقطة اللي بتميز المرشح: إن الجودة الـ acoustic مش كفاية، لازم تقيس جودة الـ transcript كمان (alignment score، نسبة الـ characters غير العربية، طول الـ transcript مقارنة بطول الصوت)، وتراجع sample عشوائي بالودن قبل ما تصدق أي metric، وتعمل dashboard لتوزيعات الـ metrics دي لكل مصدر data عشان تعرف مين المصدر اللي بيبوظ الـ training.

**سؤال متابعة:** الـ DNSMOS بيدي قيم عالية لصوت TTS صناعي ممتاز. لو بتفلتر found data بيه، ايه الخطر؟ (هتحتفظ بالصوت الصناعي اللي ممكن يكون مغذّي في الـ dataset وتعلّم الموديل artifacts).

---

## 2. الـ Speech-to-Text (ASR)

### س12. اشرح CTC وRNN-T (Transducer) وAttention Encoder-Decoder. امتى تستخدم كل واحد؟

**الإجابة النموذجية:**
الثلاثة طرق مختلفة لحل نفس المشكلة: الـ input frames أكتر بكتير من الـ output tokens ومفيش alignment معروف.

**الـ CTC:** الـ encoder بيطلع توزيع على الـ vocabulary زائد blank token لكل frame، والـ loss بتجمع احتمال كل الـ alignments اللي بتـ collapse للـ target (بالـ dynamic programming). مميزاته: بسيط، سريع جدًا في الـ inference (greedy decoding مجرد argmax وإزالة التكرار والـ blank)، streaming طبيعي لو الـ encoder causal. عيبه: الـ conditional independence assumption، كل frame بيتوقع مستقل، فمفيش language model داخلي، وعشان كده بيستفيد جدًا من external LM (n-gram أو neural) في الـ decoding.

**الـ RNN-T / Transducer:** بيضيف prediction network (زي LM صغير على الـ tokens اللي طلعت) وjoint network بيدمج الـ encoder output مع الـ prediction output. بيحل مشكلة الـ conditional independence، streaming-native، وبقى الـ standard في الـ on-device والـ streaming (Google وApple وNVIDIA Parakeet). عيبه: الـ training أثقل في الـ memory (الـ lattice بحجم T×U×V)، وفيه طرق زي pruned RNN-T في k2 وTDT (Token-and-Duration Transducer) اللي بتسرّع الـ decoding بتوقع كام frame يعدي.

**الـ AED (Attention Encoder-Decoder):** الـ decoder autoregressive بيعمل attention على الـ encoder output كله، زي Whisper. أعلى accuracy في الـ offline، وبيتعلم punctuation وcasing وحتى translation، بس مش streaming بطبيعته، وعرضة للـ hallucination والـ repetition لأن الـ decoder ممكن يتجاهل الـ audio.

الاختيار: streaming بـ latency واطية، يبقى RNN-T أو CTC مع chunked encoder. Offline batch transcription وأعلى دقة، يبقى AED أو hybrid CTC/attention (ESPnet، WeNet) اللي بيستخدم CTC للـ alignment والـ attention للـ rescoring.

**سؤال متابعة:** ليه CTC بيحتاج الـ input يكون أطول من الـ output؟ وايه اللي يحصل لو عندك token مكرر زي "لل"؟

### س13. Whisper: اشرح المعمارية، نقط الضعف المعروفة، وازاي تعمله productionize.

**الإجابة النموذجية:**
Whisper موديل encoder-decoder transformer متدرب على 680 ألف ساعة weakly supervised من الإنترنت (وlarge-v3 على أكتر مع pseudo-labels). الـ input log-mel بـ 80 filter (128 في large-v3) لـ 30 ثانية ثابتة، بيتعمله padding لو أقصر. الـ decoder بيستخدم special tokens للغة والمهمة (transcribe/translate) والـ timestamps، وبيقدر ياخد prompt نصي.

نقط الضعف: (1) الـ hallucination: على الصمت أو الضوضاء بيخترع جمل أو يكرر عبارة، وده أخطر حاجة في الـ production. (2) مش streaming، والـ 30 ثانية ثابتة بتضيّع compute على الـ padding. (3) الـ timestamps تقريبية. (4) في العربي: بيميل للفصحى، ضعيف في اللهجات والـ code-switching، وبيغلط في الهمزة والتاء المربوطة بشكل غير ثابت. (5) حساس للـ prompt وممكن يعمل loops مع condition_on_previous_text.

الـ productionization: استخدم faster-whisper (CTranslate2) بـ int8 أو fp16 مع batched inference، أو TensorRT-LLM/vLLM لو محتاج throughput أعلى. قبل الموديل: VAD (Silero) عشان تقطع الصمت ومتبعتش chunks فاضية، وده بيقلل الـ hallucination بشكل كبير. اضبط الـ thresholds: compression_ratio_threshold وno_speech_threshold وlogprob_threshold، واطفي الـ temperature fallback أو قللها لو الـ latency مهمة، واطفي condition_on_previous_text في الـ noisy audio. للـ word timestamps الدقيقة استخدم forced alignment بموديل wav2vec2 (زي whisperX) بدل timestamps الموديل. للـ streaming فيه approach الـ LocalAgreement (whisper_streaming) لكنه بيضيف latency ثانية أو أكتر، فلو الـ streaming هو الـ core use case الأنسب Conformer CTC/RNN-T. وأخيرًا fine-tuning بـ LoRA أو full FT على dialect data مع mix من الفصحى عشان متنساش.

**سؤال متابعة:** ليه large-v3-turbo أسرع؟ (الـ decoder اتقلص لـ 4 layers بدل 32، والـ encoder زي ما هو، فالـ speedup في الـ decoding مش الـ encoding). وليه distil-whisper مش مفيد في العربي؟ (اتعمل distillation على English بس).

### س14. الـ self-supervised speech models: wav2vec 2.0 وHuBERT وWavLM وw2v-BERT. الفرق بينهم وليه مهمين للعربي؟

**الإجابة النموذجية:**
كلهم بيتعلموا representations من audio من غير transcripts، والفرق في الـ pretext task:

- **الـ wav2vec 2.0:** CNN encoder على الـ raw waveform، ثم masking لجزء من الـ latents، وcontrastive loss: الموديل لازم يميز الـ quantized latent الصح من distractors.
- **الـ HuBERT:** بدل الـ contrastive، بيتوقع cluster IDs جاية من k-means على MFCC في الأول، وبعدين على الـ hidden states بتاعة الموديل نفسه (iterative refinement). أبسط وأكثر استقرارًا، والـ units بتاعته بقت أساس الـ semantic tokens في الـ speech LMs.
- **الـ WavLM:** زي HuBERT مع إضافة denoising وoverlap simulation في الـ pretraining، فبقى أقوى في speaker tasks وdiarization، وهو الـ backbone المفضل للـ speaker similarity metrics.
- **الـ w2v-BERT:** بيدمج contrastive وmasked prediction في نفس الموديل، وw2v-BERT 2.0 اللي في Seamless من Meta متدرب على 4.5 مليون ساعة multilingual وبيعتبر من أقوى نقط البداية للغات قليلة الموارد.
- **الـ XLS-R وMMS:** نسخ multilingual (MMS بيغطي 1000+ لغة) مناسبة كنقطة بداية.

ليه مهمين للعربي؟ لأن اللهجات low-resource في الـ labeled data، لكن الـ unlabeled audio متوفر (راديو، بودكاست، مكالمات). الـ recipe: continued pretraining على unlabeled dialect audio لو عندك آلاف الساعات، ثم fine-tuning بـ CTC head على مئات الساعات labeled. بضع مئات الساعات مع SSL backbone كويس ممكن تتفوق على Whisper في dialect محدد. كمان الـ SSL features بتتستخدم في TTS evaluation وفي بناء tokens للـ speech LLMs.

**سؤال متابعة:** ايه الفرق بين استخدام الـ SSL model كـ frozen feature extractor وfine-tuning كامل؟ وأنهي layers بتاخد منها الـ features للـ phonetic content مقابل الـ speaker identity؟ (الـ phonetic content في الـ layers الوسطى والعليا، والـ speaker في الـ layers الأولى غالبًا).

### س15. الـ streaming STT: ازاي تخلي الـ encoder streamable؟ اشرح chunked attention والـ lookahead والـ latency-accuracy tradeoff.

**الإجابة النموذجية:**
الـ Transformer/Conformer encoder العادي بيشوف الـ utterance كله (full context)، فمينفعش في الـ streaming. الحلول:

1. **الـ causal encoder:** كل frame بيشوف اللي قبله بس. أقل latency لكن أسوأ accuracy لأن الـ right context مهم جدًا للـ phonetics (الحرف بيتحدد أحيانًا باللي بعده).
2. **الـ chunked attention:** بتقسم الـ input لـ chunks (مثلًا 320 أو 640 ms)، والـ attention جوه الـ chunk full، ومع الـ chunks اللي قبله left context محدود. الـ algorithmic latency هي حجم الـ chunk تقريبًا.
3. **الـ lookahead / right context:** بتسمح لكل chunk يشوف كام frame من الـ chunk اللي بعده، بيحسّن الـ accuracy بس بيضيف latency بنفس المقدار.
4. **الـ causal convolutions** في الـ Conformer بدل الـ symmetric ones.
5. **الـ dynamic chunk training** (WeNet): بتدرب بأحجام chunks عشوائية، فموديل واحد يشتغل بأكتر من latency setting وكمان offline. وU2/U2++: CTC للـ first pass streaming والـ attention decoder للـ rescoring في نهاية الـ utterance.

الـ tradeoff الحقيقي: chunk أكبر = accuracy أعلى وlatency أعلى. في الـ voice agents، بنستهدف emission latency (الفرق بين وقت نطق الكلمة ووقت ظهورها) في حدود 200 إلى 400 ms. كمان لازم يفرّق بين الـ partial results (اللي ممكن تتغير) والـ final results، وإن استقرار الـ partials مهم للـ UX ولو هتبني speculative LLM calls عليها.

**سؤال متابعة:** ازاي تقيس الـ emission latency عمليًا؟ (word timestamps من forced alignment مقابل وقت emission بتاع الـ token في الـ log).

### س16. الـ decoding: greedy مقابل beam search، والـ LM fusion، والـ contextual biasing للـ hotwords.

**الإجابة النموذجية:**
في CTC الـ greedy decoding مجرد argmax لكل frame ثم collapse، سريع لكن بيتجاهل الـ prefix probability. الـ prefix beam search بيجمع كل الـ alignments اللي بتوصل لنفس الـ prefix، وبيسمح بإضافة LM: score = log P_acoustic + λ log P_LM + β × word_count. مع n-gram KenLM (بـ pyctcdecode مثلًا) الـ CTC model بيحصل على تحسن كبير في WER، خصوصًا في الـ domain-specific text. ده shallow fusion. في الـ RNN-T فيه internal LM بالفعل، فلما تضيف external LM لازم تطرح تقدير الـ internal LM (ILME) وإلا الـ bias بيتضاعف. الـ WFST decoding (TLG في Kaldi وk2) بيدمج tokens وlexicon وgrammar في graph واحد وبيدي تحكم كامل.

الـ contextual biasing: لما يكون عندك أسماء منتجات أو أماكن سعودية أو مصطلحات بنكية نادرة في الـ training data. الطرق: (1) boosting وقت الـ decoding بـ prefix tree على الـ hotwords (بتضيف bonus للـ paths اللي بتمشي على الـ hotword). (2) deep contextual biasing: encoder للـ bias phrases والموديل بيعمل attention عليها (CLAS وما بعده). (3) في Whisper، الـ initial prompt بيشتغل كـ biasing ضعيف ومش مضمون. (4) fine-tuning بـ synthetic TTS data للمصطلحات دي.

**سؤال متابعة:** ليه الـ LM weight λ لازم يتعمله tuning على dev set، وايه اللي يحصل لو كبير أوي؟ (الموديل بيعمل "تصحيح" لكلام سليم وبيقلل الـ rare words).

### س17. الـ Arabic STT: الـ text normalization، وازاي تحسب WER بشكل عادل، وامتى تستخدم CER؟

**الإجابة النموذجية:**
الـ WER في العربي بيتضخم من اختلافات إملائية مش أخطاء حقيقية: أشكال الهمزة (أ إ آ ا)، ة/ه في الآخر، ى/ي، التشكيل (اللي غالبًا بيتشال من الـ transcripts)، الأرقام (٢٥ مقابل 25 مقابل "خمسة وعشرين")، والمسافات في الكلمات المركبة. في اللهجات مفيش orthography موحدة أصلًا (كده/كدة/كدا، ليش/ليه)، والـ code-switching ممكن يتكتب بحروف لاتينية أو عربية (meeting مقابل ميتنج).

عشان تقيس عدل: طبّق نفس الـ normalizer على الـ reference والـ hypothesis: شيل التشكيل والتطويل، وحّد الهمزات والتاء المربوطة والألف المقصورة (أو خليهم لو الـ test set مضبوط لغويًا وانت عايز تعاقب الأخطاء دي)، وحّد الأرقام بسياسة واحدة (ITN للاتنين أو العكس)، وحّد الـ script للـ code-switching، وشيل الترقيم. وسجّل نسختين من الـ WER: raw وnormalized، عشان الشفافية.

الـ CER (Character Error Rate) مفيد لأن العربي agglutinative: "وبالسيارة" كلمة واحدة، وغلطة في الـ clitic بتخلي الكلمة كلها غلط في الـ WER رغم إن 90% منها صح. فبنقرر الاتنين، والـ CER أنسب للمقارنة بين موديلات في اللهجات.

الـ test sets: لازم تكون dialect-specific ومطابقة للـ channel (telephony مقابل app). للسعودية فيه SADA، ولعموم العربي MGB-2 (فصحى) وMGB-3 (مصري) وMGB-5 (مغربي) وCasablanca (لهجات متعددة)، لكن الأفضل test set داخلي من نفس الـ domain. ولازم تعرف الـ inter-annotator agreement على اللهجة، لأنه بيحدد الـ floor: لو اتنين annotators بيختلفوا 8%، الموديل مش هيوصل أقل من كده بشكل ذو معنى.

**سؤال متابعة:** ازاي تتعامل مع الأرقام في الـ reference؟ وايه الفرق بين WER كلي وWER per-dialect لما تبلّغ نتائج للإدارة؟

### س18. الـ speaker diarization: ازاي بتشتغل، والفرق بين الـ pipeline التقليدي وEEND، وازاي تدمجها مع الـ STT؟

**الإجابة النموذجية:**
الـ pipeline التقليدي (زي pyannote): (1) segmentation/VAD، (2) استخراج speaker embeddings (ECAPA-TDNN أو WavLM-based) من windows قصيرة، (3) clustering (spectral، agglomerative، أو VBx) لتحديد عدد المتكلمين وتجميع الـ segments، (4) resegmentation لتنعيم الحدود. المشكلة الأساسية: الـ overlap (اتنين بيتكلموا مع بعض) والـ turns القصيرة جدًا. الـ EEND (End-to-End Neural Diarization) بيتوقع لكل frame احتمال كل متكلم مباشرة بـ permutation-invariant training، فبيتعامل مع الـ overlap طبيعي، بس محدود في عدد المتكلمين وبيحتاج data كتير. الـ hybrid (EEND للـ local segments + clustering للـ global) هو الـ state of the art عمليًا، وpyannote 3.x بيمشي في الاتجاه ده.

الـ metric: DER = (missed speech + false alarm + speaker confusion) / total speech، مع collar عادةً 250 ms.

الدمج مع STT: تعمل STT بـ word-level timestamps (forced alignment)، وتعمل diarization منفصلة، وتربط كل كلمة بالمتكلم اللي بيغطي الـ timestamp بتاعها (زي whisperX). البديل الأحدث: joint models بتطلع speaker tokens مع الـ transcript. النصيحة العملية اللي لازم تسمعها: في الـ call center لو التسجيل stereo (قناة لكل طرف) استخدم channel separation وبلاش diarization خالص، هتكسب دقة وتوفر compute.

**سؤال متابعة:** ليه الـ speaker embeddings المدربة على VoxCeleb بتشتغل كويس على العربي؟ (الـ speaker identity مش language-dependent بشكل كبير، لكن الـ channel mismatch أهم من اللغة).

### س19. الـ forced alignment والـ word timestamps: MFA وCTC forced alignment وWhisper timestamps. ليه محتاجهم؟

**الإجابة النموذجية:**
الـ forced alignment بياخد audio ونص معروف ويطلع وقت كل كلمة أو phoneme. الطرق: (1) MFA (Montreal Forced Aligner): GMM-HMM كلاسيكي مع pronunciation dictionary، دقيق جدًا على مستوى الـ phoneme، بس محتاج lexicon/G2P عربي، وده مشكلة مع نص غير مشكّل. (2) CTC forced alignment (torchaudio forced_align مع wav2vec2 أو MMS aligner): بتعمل Viterbi على الـ CTC emissions مقيدة بالنص، سريع وبيشتغل على character level من غير lexicon، ومناسب للعربي. (3) Whisper timestamps من الـ cross-attention weights بـ DTW: تقريبية وبتتأثر بالـ hallucination.

الاستخدامات: تقطيع تسجيلات طويلة لـ utterances قصيرة للتدريب (TTS وSTT)، تنظيف الـ data (لو الـ alignment score واطي غالبًا الـ transcript غلط)، subtitles، ربط الـ diarization بالكلمات، قياس الـ emission latency، وفي TTS القديم (FastSpeech 2) كانت الـ durations نفسها بتيجي من MFA.

**سؤال متابعة:** ازاي تستخدم الـ alignment score كـ filter لتنظيف dataset فيها transcripts مش دقيقة؟

### س20. عندك 200 ساعة labeled من لهجة سعودية. ازاي تعمل fine-tuning لموديل STT؟ الـ data والـ augmentation والـ hyperparameters والـ evaluation.

**الإجابة النموذجية:**
**اختيار الـ base model:** حسب الـ use case: لو offline ومحتاج punctuation، Whisper large-v3 أو turbo. لو streaming، Conformer/FastConformer RNN-T أو CTC من NeMo، أو w2v-BERT 2.0/MMS بـ CTC head. الأهم إن الـ tokenizer يغطي العربي كويس (Whisper tokenizer بيقطع العربي لـ tokens كتير، وده بيطوّل الـ sequence ويبطئ الـ decoding).

**الـ data قبل أي حاجة:** وحّد الـ transcripts بـ normalizer واحد (نفس اللي هتقيّم بيه)، اعمل forced alignment وارمي الـ segments اللي الـ score بتاعها واطي، dedupe، اقطع لـ 3 إلى 20 ثانية، واتأكد إن الـ speakers في الـ test set مش موجودين في الـ train. لو فيه كلام فصحى مع اللهجة، خليه، لأن الواقع كده.

**الـ augmentation:** SpecAugment (time/frequency masking)، speed perturbation (0.9/1.0/1.1)، إضافة noise من MUSAN وroom impulse responses، محاكاة الـ telephony codec لو الـ target تليفون، volume perturbation. ده بيعوّض جزء كبير من قلة الـ data.

**الـ training:** في Whisper: learning rate صغير (1e-5 أو أقل)، warmup، وممكن LoRA على الـ decoder بس أو الاتنين، مع mixing لـ MSA data (replay) عشان تتجنب الـ catastrophic forgetting للفصحى. في الـ Conformer: full fine-tuning عادي بـ lr أعلى. Early stopping على dev WER مش على الـ loss. Curriculum: ابدأ بالـ clean segments.

**الـ evaluation:** held-out speakers، WER وCER بعد normalization، breakdown حسب اللهجة والـ channel والـ noise level، regression test على MSA set عشان تتأكد إنك مخسرتش الفصحى، فحص الـ hallucination على silence samples، وعينة يسمعها لغوي.

**التوسع بعد كده:** pseudo-labeling على unlabeled audio بالموديل الجديد مع filtering بالـ confidence أو بالاتفاق بين موديلين (noisy student)، وده غالبًا بيدي تحسن أكبر من أي tweak في الـ hyperparameters.

**سؤال متابعة:** لو الـ WER على الـ dev نزل بس الـ WER على الفصحى طلع، ايه القرار؟ ولو 30% من الـ data من نفس المتكلم، ايه الخطر؟

### س21. الـ ITN (Inverse Text Normalization) والـ punctuation restoration: ايه هما، فين مكانهم في الـ pipeline، وازاي في العربي؟

**الإجابة النموذجية:**
الـ ITN بيحوّل الـ spoken form للـ written form: "خمسة وعشرين ريال وخمسين هللة" تبقى "25.50 ريال"، "تسعة أبريل" تبقى "9 أبريل"، وكذلك أرقام التليفون والوقت والنسب. مهم لأن الـ LLM اللي بعد الـ STT والـ tools (بحث عن رقم حساب مثلًا) محتاجين الشكل المكتوب، وكمان للعرض للمستخدم. الطرق: WFST grammars (زي اللي في NeMo text processing) rule-based وdeterministic ومناسبة للـ production، أو seq2seq/LLM-based أكثر مرونة لكن ممكن يهلوس. في العربي الـ grammars محتاجة تتعامل مع أشكال الأرقام حسب الجنس والحالة الإعرابية (ثلاثة وثلاث) واللهجات (تلاتة، ثلاث) والأرقام الهندية والعربية.

الـ punctuation restoration: موديلات CTC/RNN-T ما بتطلعش ترقيم، فبتحتاج موديل منفصل (token classification على BERT عربي) يضيف نقطة وفاصلة وعلامة استفهام. Whisper بيطلع ترقيم لكن غير ثابت في العربي. الترقيم مهم للـ LLM prompt وللـ sentence splitting قبل الـ TTS.

المكان في الـ pipeline: STT raw ثم punctuation ثم ITN ثم LLM. وفي الـ evaluation لازم تحدد هل الـ WER بيتحسب قبل ولا بعد الـ ITN، لأن الفرق كبير.

**سؤال متابعة:** ايه الفرق بين TN (للـ TTS) وITN (للـ STT)، وليه محتاج الاتنين يكونوا متسقين في voice agent واحد؟

### س22. قارن بين الـ Conformer والـ FastConformer والـ Zipformer والـ E-Branchformer كـ encoders للـ STT. ايه اللي كل واحد غيّره وليه؟

**الإجابة النموذجية:**
الـ Conformer (2020) جمع الـ self-attention (بيلقط الـ global context) مع convolution module (بيلقط الـ local patterns زي الـ formant transitions) في block واحد بترتيب macaron: نص FFN، ثم MHSA بـ relative positional encoding، ثم conv module (pointwise conv وGLU وdepthwise conv وBatchNorm وSwish)، ثم نص FFN تاني. قبل الـ blocks فيه conv subsampling 4x بيوصل الـ frame rate لـ 40 ms. بقى الـ default في ESPnet وNeMo وWeNet لسنين.

الـ FastConformer (NVIDIA) غيّر الـ subsampling لـ 8x باستخدام depthwise separable convolutions، فالـ sequence بقت نصها، والـ attention (اللي تكلفته تربيعية) بقى أرخص 4 مرات تقريبًا، وده اللي خلى موديلات زي Parakeet وCanary أسرع بكتير من Whisper في الـ inference مع دقة مماثلة أو أحسن على الإنجليزي. كمان بيسمح بـ long-form inference بـ limited context attention.

الـ Zipformer (من فريق k2/icefall) اتعامل مع الـ encoder كـ U-Net: stacks بتشتغل على frame rates مختلفة (50 ثم 25 ثم 12.5 Hz وأقل) وترجع تتدمج، مع إعادة استخدام الـ attention weights بين modules، وتفاصيل تدريب زي BiasNorm وScaledAdam وEden scheduler. بيدي دقة عالية بـ parameters وcompute أقل، بس الـ implementation أعقد وأصعب في الـ export.

الـ E-Branchformer بيحط الـ attention وcgMLP (convolutional gating MLP) في فرعين متوازيين وبيدمجهم، وأثبت في ESPnet وOWSM إنه أحسن شوية من الـ Conformer مع تدريب أكثر استقرارًا.

اللي بيفرق: المرشح يعرف إن الاختيار بيتحكم فيه الـ ecosystem والـ export path (NeMo وRiva للـ FastConformer، sherpa للـ Zipformer)، والـ streaming support (كلهم ينفع يبقوا chunked لكن الـ recipes الجاهزة بتختلف)، والـ compute في الـ inference أكتر من فرق الدقة الصغير.

**سؤال متابعة:** ليه الـ BatchNorm جوه الـ conv module بيعمل مشاكل في الـ streaming والـ batch sizes الصغيرة، وايه البديل؟ (LayerNorm أو GroupNorm، أو تجميد إحصائيات الـ BN).

### س23. الـ tokenization في الـ STT: characters مقابل BPE/SentencePiece، حجم الـ vocab، والخصوصيات العربية. وايه علاقة الـ tokenizer بالـ CTC والـ RNN-T؟

**الإجابة النموذجية:**
الـ character-level vocab صغير (للعربي حوالي 40 حرف زائد أرقام وعلامات)، وبيخلي الموديل يقدر يكتب أي كلمة، بس الـ sequences أطول وبيحتاج الـ encoder يشوف context أطول عشان يفك الكلمة. الـ BPE (SentencePiece unigram أو BPE) بيقصّر الـ sequence وبيدي الموديل "ذاكرة لغوية" ضمنية، وبيحسّن الـ CTC بالذات لأن الـ CTC عنده conditional independence بين الـ frames فكل ما الـ token يحمل معلومات أكتر كل ما التوقع أسهل. الأحجام المعتادة 128 لـ 1024 للـ CTC وRNN-T، وأكبر بكتير في الـ AED (Whisper بيستخدم tokenizer نصي عام حوالي 50 ألف token، وده جزء من سبب قوته اللغوية وكمان سبب الـ hallucination).

القيد في الـ CTC: طول الـ output ما ينفعش يزيد عن عدد الـ frames بعد الـ subsampling. مع 8x subsampling (80 ms) والكلام السريع، الـ character-level ممكن يكسر القيد ده، والـ BPE بيحله. وفي الـ CTC الـ blank هو اللي بيفصل الـ tokens المتكررة (لو كتبت "الله" بحرفين لام متتاليين لازم blank بينهم)، فالـ tokenizer بيأثر على توزيع الـ blanks.

الخصوصيات العربية: (1) هل التشكيل جزء من الـ vocab؟ لو الـ transcripts مش مشكّلة بشكل متسق، الأفضل إزالته من التدريب، لأن الموديل هيتعلم توزيع عشوائي. (2) الـ SentencePiece لازم يتدرب على النص المطبّع (normalized) بنفس قواعد الإنتاج. (3) الـ code-switching: لو الإنجليزي هيتكتب بحروف لاتينية، الـ vocab لازم يشمل الاتنين والـ BPE يتدرب على mixed corpus بنسب تمثل الواقع. (4) الـ byte fallback (في SentencePiece) بيمنع الـ unknown tokens لما يظهر رمز غريب.

**سؤال متابعة:** غيّرت الـ tokenizer لموديل موجود. ايه اللي تقدر تحتفظ به من الـ weights وايه اللي لازم يتدرب من الأول؟ (الـ encoder ينفع، الـ output layer والـ prediction network/decoder embeddings لأ).

### س24. تدريب الـ RNN-T مكلف في الذاكرة. اشرح ليه، والحلول: pruned RNN-T، والـ TDT، والـ multi-blank، والـ HAT.

**الإجابة النموذجية:**
الـ joint network في الـ transducer بيحسب logits لكل زوج (frame t، label position u)، فالـ tensor حجمه T × U × V. لـ utterance 20 ثانية (500 frame عند 40 ms) بـ 100 token وvocab 1024، ده 51 مليون قيمة لكل utterance في الـ batch، ومع batch 32 وfp32 بتوصل لجيجابايتات قبل ما تحسب الـ loss. ده اللي كان بيخلي الـ transducers تتدرب بـ batches صغيرة أو على GPUs كبيرة.

الحلول: (1) الـ pruned RNN-T (من k2): بيحسب أول joiner بسيط (مجرد جمع للـ encoder والـ predictor outputs) بيطلع منه bounds للـ alignment، وبعدين بيحسب الـ joiner الكامل بس في شريط ضيق حوالين الـ alignment المتوقع (مثلًا 5 tokens لكل frame)، فالتكلفة بتنزل من T × U لـ T × S. (2) function merging وfused kernels (warp-transducer وNeMo numba loss) بتقلل الـ intermediate tensors. (3) الـ TDT (token-and-duration transducer) بيخلي الموديل يتوقع كمان كام frame يقفز بعد كل token، فبيقلل عدد خطوات الـ decoding جدًا (أسرع في الـ inference) ومن غير ما يخسر دقة. (4) الـ multi-blank transducer بيضيف blanks بتقفز أكتر من frame. (5) الـ HAT (hybrid autoregressive transducer) بيفصل احتمال الـ blank عن توزيع الـ labels، وده بيسمح بتقدير الـ internal LM بشكل صحيح للـ ILME لما تدمج LM خارجي.

المرشح القوي بيذكر كمان: الـ prediction network ممكن يكون stateless (يشوف آخر token أو اتنين بس) من غير خسارة كبيرة، وده بيقلل الـ overfitting على النص ويسهّل الـ LM fusion.

**سؤال متابعة:** ليه الـ TDT بيدي تسريع أكبر في الـ inference من الـ pruned loss اللي بيسرّع التدريب بس؟

### س25. الـ hybrid CTC/attention والـ intermediate CTC والـ self-conditioned CTC: ليه بنضيف CTC loss لموديل attention، وازاي الـ joint decoding بيشتغل؟

**الإجابة النموذجية:**
الـ attention decoder حر في الـ alignment، وده بيخليه يتعلم ببطء في البداية وأحيانًا يتعلم alignments غلط (يتخطى أو يكرر). الـ CTC بالعكس مقيّد بـ monotonic alignment. لما تضيف CTC head على الـ encoder وتدرب بـ loss مركّب (مثلًا 0.3 CTC و0.7 attention)، الـ encoder بيتعلم alignment صح بدري، والتدريب بيستقر، والنتيجة أحسن من أي واحد لوحده. ده الـ default في ESPnet من 2017.

في الـ decoding، الـ joint CTC/attention beam search بيستخدم الـ CTC prefix scores عشان يعاقب الـ hypotheses اللي الـ attention بيحبها بس مش متسقة مع الصوت (زي الـ hallucination والتكرار)، وبيوقف الـ decoding في مكان معقول بدل ما الـ attention يستمر. التكلفة إن الـ CTC prefix scoring مكلف حسابيًا، فبعض الأنظمة بتستخدم الـ CTC بس للـ endpoint detection أو للـ rescoring.

الـ intermediate CTC بيحط CTC loss على layers وسطى في الـ encoder (مثلًا layer 6 من 12) كـ regularization، والـ self-conditioned CTC بيرجّع توقعات الـ layer الوسطى للـ layers اللي بعدها، فالـ layers العليا بتشوف "مسودة" من النص وبتصحّحها، وده بيكسر جزئيًا الـ conditional independence بتاعة الـ CTC وبيحسّن الـ CTC-only models بشكل ملموس من غير decoder autoregressive. عمليًا ده بيخلي CTC model سريع جدًا يقرب من دقة الـ transducer.

**سؤال متابعة:** موديل CTC-only بيطلع WER أعلى من الـ RNN-T بنسبة 10% نسبيًا على نفس الـ data. ايه الحاجات اللي تجربها قبل ما تنقل لـ transducer؟ (intermediate/self-conditioned CTC، BPE أكبر، LM fusion، encoder أكبر).

### س26. تدريب موديل STT متعدد اللغات والمهام: الـ language وtask tokens، الـ sampling بين اللغات، والـ catastrophic forgetting. ايه اللي Whisper عمله وايه اللي تعمله مختلف للعربي؟

**الإجابة النموذجية:**
Whisper بيحط في بداية الـ decoder sequence مجموعة special tokens: token اللغة، ثم المهمة (transcribe أو translate)، ثم هل فيه timestamps ولا لأ، وقبلهم اختياريًا previous text كـ prompt. الـ decoder بيتعلم يكيّف نفسه على اللغة من الـ token ده، وفي الـ inference لو ما حددتش اللغة الموديل بيتوقعها من أول 30 ثانية. المزج ده خلى موديل واحد يخدم 99 لغة، لكن التوزيع كان غير متوازن جدًا: الإنجليزي أغلب الـ data، والعربي كام ألف ساعة بس، وأغلبها MSA من مصادر إعلامية.

في الـ multilingual training فيه tradeoff بين اللغات: الـ sampling بالتناسب مع حجم الـ data بيهمل اللغات الصغيرة، والـ uniform sampling بيضر اللغات الكبيرة ويعمل overfitting على الصغيرة. الحل المعتاد temperature sampling (الاحتمال يتناسب مع حجم الـ data أس 1/T، بـ T حوالي 3 إلى 5). وبتظهر ظاهرة الـ positive transfer بين اللغات القريبة (العربي بيستفيد من الفارسي والأردو في الـ script جزئيًا) والـ negative interference لما الموديل صغير.

للعربي وللهجاته، النقاش الأهم: هل تعمل token لكل لهجة؟ الرأي العملي: token واحد للعربي مع dialect token اختياري (أو prompt) لأن الحدود بين اللهجات مش حادة والمتكلم بيتنقل بين MSA واللهجة في نفس الجملة، ولو خليت الـ dialect token إلزامي الموديل هيتوهم في الـ inference لما تديله token غلط. وعن الـ forgetting: لما تعمل fine-tuning على لهجة، الموديل بيخسر MSA وباقي اللغات بسرعة، والحل replay (خلط 10-30% من الـ data الأصلية أو data متنوعة) أو LoRA أو تجميد الـ encoder جزئيًا، والأهم إنك تقيس على test sets للمهام اللي مش بتدربها.

**سؤال متابعة:** لو عايز الموديل يطلّع الإنجليزي بحروف لاتينية جوه الجملة العربية، ايه اللي لازم يتغير في الـ data والـ tokens؟

### س27. الـ long-form transcription (ساعة أو أكتر): ازاي الـ AED models بتتعامل مع طول الـ context، وايه الفرق بين sequential وchunked وVAD-based approaches، وايه مشاكل الـ timestamps drift؟

**الإجابة النموذجية:**
Whisper بيشوف 30 ثانية بس في المرة. للملفات الطويلة فيه ثلاث طرق: (1) الـ sequential decoding الأصلي: الموديل بيطلّع timestamps، وبتقص الـ window التالية من آخر timestamp، وبتمرر النص السابق كـ prompt (condition_on_previous_text). دقيق في السياق لكنه بطيء (مفيش batching) وعرضة لـ repetition loops: لو الموديل غلط مرة، الغلط بيتغذى في الـ prompt وبيتكرر لدقايق. (2) الـ chunked approach (HF pipeline): تقطيع بـ overlap (مثلًا 30 ثانية بـ stride 5) وdecoding متوازي، ثم دمج الـ boundaries بمطابقة الـ tokens. أسرع بكتير بس ممكن يعمل أخطاء عند الحدود ويفقد السياق. (3) الـ VAD-based (WhisperX): تقطيع عند الصمت لـ segments أقل من 30 ثانية، وdecoding في batches. ده الأنسب عمليًا لأنه بيتجنب القطع في نص كلمة وبيلغي الـ hallucination على الصمت.

الـ encoder-only models (CTC/RNN-T) ما عندهاش حد الـ 30 ثانية نظريًا لأن الـ attention ممكن يكون limited-context أو الـ conv local، فبتعمل buffered inference بـ chunks طويلة (مثلًا 20-30 ثانية مع context جانبي) ودمج على مستوى الـ frames، وده أنضف من الـ AED. Canary وParakeet في NeMo عندهم recipes جاهزة لكده.

الـ timestamps drift: مع الـ sequential decoding، الـ timestamps ممكن تتزحزح تدريجيًا لو الموديل قدّر آخر timestamp غلط، أو لما يكون فيه موسيقى أو صمت طويل. الحل: تثبيت الـ timestamps على الـ VAD boundaries (اللي جاية من الـ audio مش من الموديل)، واستخدام forced alignment للـ word-level بدل timestamps الموديل.

**سؤال متابعة:** عندك بودكاست 3 ساعات بمتكلمين وموسيقى بين الفقرات. ايه الـ pipeline كامل من الملف الخام للـ transcript بـ timestamps ومتكلمين؟

### س28. الـ hallucination والـ confidence estimation في الـ STT: ليه الـ AED بيهلوس، وازاي تكتشف الهلوسة، وازاي تحسب confidence موثوق للـ CTC والـ RNN-T والـ AED؟

**الإجابة النموذجية:**
الـ AED decoder هو language model قوي متعلّم على النص، ولما يكون الصوت ضعيف الإشارة (صمت، ضوضاء، لغة مش موجودة في التدريب) الـ decoder بيكمّل من الـ prior اللغوي بدل الصوت: جمل كاملة مش موجودة، تكرار، ترجمة بدل نسخ، أو نصوص شهيرة من الـ training data (زي "اشترك في القناة"). Whisper بالذات بيهلوس على الصمت لأن الـ training data اليوتيوبية فيها subtitles على مقاطع صامتة.

الاكتشاف: (1) الـ VAD قبل الموديل (أهم حاجة). (2) الـ compression ratio للنص (لو النص بيتضغط بـ gzip لأقل من نصه فغالبًا تكرار). (3) الـ average log-probability للـ segment وthreshold عليه. (4) الـ no_speech probability. (5) مقارنة طول النص بطول الصوت (كلمات كتير في ثانيتين = هلوسة). (6) في الـ production: الـ temperature fallback مش مناسب للـ real-time، فبتعتمد على الـ VAD والـ thresholds وتبديل الموديل لـ CTC/RNN-T اللي بيهلوس أقل بكتير لأنه مقيّد بالـ alignment.

الـ confidence: في الـ CTC، الـ posterior بتاع الـ token بعد تجميع الـ frames المتكررة (مثلًا max أو mean على الـ frames اللي طلّعت نفس الـ token)، وفيه طرق بتستخدم الـ entropy بدل الـ max probability لأنها أحسن calibration، وNeMo عنده confidence estimation module بيطبق ده على الـ CTC والـ transducer. في الـ AED، احتمال الـ token اللي اتولّد مش موثوق (الموديل واثق حتى وهو بيهلوس)، فبتستخدم الـ attention entropy أو agreement بين الـ hypotheses أو موديل CTC مساعد. وفي كل الحالات لازم تقيس الـ calibration (ECE ورسم reliability diagram) على data من الـ production، وتستخدم الـ confidence لحاجة عملية: توجيه للمراجعة البشرية، طلب إعادة، أو تحديد الـ pseudo-labels اللي تدخل التدريب.

**سؤال متابعة:** الـ confidence بتاع الموديل بعد الـ fine-tuning بقى overconfident على اللهجة الجديدة. ازاي تعمل recalibration من غير retraining؟ (temperature scaling على dev set).

### س29. الـ code-switching العربي-الإنجليزي في الـ STT: ايه الصعوبات في الـ data والـ tokenization والـ evaluation، وازاي تبني نظام يتعامل معاه؟

**الإجابة النموذجية:**
في الخليج ومصر الكلام اليومي فيه مصطلحات إنجليزية كتير (أسماء منتجات، مصطلحات عمل، "OK"، "already")، وأحيانًا جمل كاملة. الصعوبات: (1) الـ script: هل "meeting" تتكتب "meeting" ولا "ميتنج"؟ لازم قرار في الـ guidelines ويتطبق باتساق، والأكثر فائدة للـ downstream غالبًا الكتابة اللاتينية للكلمات الإنجليزية الواضحة والعربية للكلمات المعرّبة المستقرة (زي "تليفون"). (2) قلة الـ data: الـ corpora العامة قليلة (ArzEn للمصري-الإنجليزي مثال)، فبتحتاج تجمع data من مكالمات حقيقية أو تصنّعها. (3) الـ language ID على مستوى الكلمة بيتلخبط في الأسماء الخاصة. (4) الموديلات الـ multilingual بتميل تختار لغة واحدة للـ segment كله (Whisper بيقرر لغة من الـ language token) فبتترجم الكلمة الإنجليزية للعربي أو العكس.

بناء النظام: (1) موديل واحد multilingual بـ vocab مشترك (BPE متدرب على mixed text) بدل موديلين مع switching. (2) data augmentation: توليد جمل مختلطة بـ LLM ثم TTS-ها بأصوات متنوعة، أو concatenation لمقاطع عربية وإنجليزية من نفس المتكلم، مع الحذر إن الـ synthetic ما يتعدّاش نسبة معينة. (3) الـ fine-tuning على data حقيقي مختلط حتى لو قليل (عشرات الساعات بتفرق). (4) في Whisper، الـ initial prompt المختلط بيدفع الموديل للكتابة المختلطة.

الـ evaluation: الـ WER العادي بيعاقب على "meeting" مقابل "ميتنج" كخطأ كامل، فبتحتاج normalization بتحوّل الاتنين لصورة موحدة (transliteration table) أو تقيس WER على كل لغة منفصلة، وتضيف metric لـ language ID accuracy، وتبني test set فيه code-switching بنسبة زي الواقع.

**سؤال متابعة:** الـ LLM اللي بعد الـ STT بيستقبل "ميتنج" أحيانًا و"meeting" أحيانًا. ده بيعمل مشاكل فين، وازاي تحلها؟

### س30. الـ speech translation: end-to-end مقابل cascade، الـ simultaneous translation، وامتى تحتاجها في السوق العربي؟

**الإجابة النموذجية:**
الـ cascade هو STT ثم MT، والـ end-to-end (AST) هو موديل بياخد صوت بلغة وبيطلّع نص بلغة تانية مباشرة (Whisper بيعمل X إلى الإنجليزي بس، وCanary وSeamlessM4T بيعملوا اتجاهات متعددة). مزايا الـ cascade: كل مكوّن بيتحسن لوحده، الـ MT بيستفيد من ملايين الجمل النصية، وعندك transcript وسيط تقدر تصلّحه وتراجعه. مزايا الـ end-to-end: مفيش error propagation من الـ STT، وبيقدر يستخدم الـ prosody في الترجمة، وlatency أقل. عمليًا في العربي، الـ cascade لسه بيكسب في أغلب الحالات لأن الـ MT العربي-الإنجليزي النصي (بما فيه الـ LLMs) أقوى بكتير من أي AST متدرب على data صوتية محدودة، وخصوصًا للهجات.

الـ simultaneous translation بتضيف قرار "أترجم دلوقتي ولا أستنى كلمة كمان" (سياسات wait-k وأشكال adaptive)، ومشكلة العربي إن ترتيب الجملة مختلف (الفعل بيجي أول في MSA والمفعول به آخر) فالانتظار مطلوب أكتر.

الـ use cases في السوق: الـ compliance والـ QA لمراكز الاتصال اللي فيها مشرفين مش متكلمين عربي، تلخيص المكالمات بالإنجليزي للـ management، الاجتماعات المختلطة، والـ dubbing. في الأغلب المطلوب مش ترجمة حرفية real-time لكن تلخيص مترجم بعد المكالمة، وده أسهل بكتير: STT جيد ثم LLM بيلخص ويترجم في خطوة واحدة.

التقييم: BLEU قديم وما بيحسش بالمعنى، COMET أحسن، والأهم human evaluation على sample، ومقياس منفصل لدقة الأسماء والأرقام لأن ده اللي بيبوظ الفائدة.

**سؤال متابعة:** الـ STT بيكتب اسم العميل "محمد" والـ MT بيترجمه أحيانًا "Mohammed" وأحيانًا "Muhammad". ازاي تثبّت ده في pipeline بنكي؟

### س31. عندك STT كويس بس الـ domain vocabulary (أسماء منتجات، مصطلحات) بيتغلط. ازاي تعمل adaptation من غير data صوتي؟ LM fusion، وTTS-generated data، والـ LLM rescoring والـ generative error correction.

**الإجابة النموذجية:**
فيه أربع مستويات بترتيب التكلفة: (1) الـ contextual biasing وقت الـ decoding: قائمة كلمات مع boost (word boosting في NeMo وRiva، prefix tree في الـ CTC beam search، initial prompt في Whisper). رخيص وفوري، بس بيشتغل بس على الكلمات اللي الموديل شبه قادر يطلّعها، ولو زودت الـ boost بيبدأ يحط الكلمات في أماكن غلط. (2) الـ external LM: n-gram (KenLM) متدرب على نصوص الـ domain ومدمج بـ shallow fusion، أو LM أكبر في الـ rescoring على N-best. بيساعد على الأنماط اللغوية العامة للـ domain مش بس الأسماء. (3) الـ TTS-generated audio: تولّد جمل الـ domain بعدة أصوات وتعمل fine-tuning عليها مع خلط data حقيقي (نسبة الـ synthetic ما تزيدش عن 20-30% تقريبًا) عشان الموديل ما يتعلمش artifacts الـ TTS. بيشتغل كويس للكلمات النادرة، وأحسن مع TTS بجودة عالية وتنوع أصوات. (4) الـ generative error correction: تبعت الـ N-best أو الـ transcript للـ LLM مع قائمة المصطلحات وتخليه يصلّح. قوي جدًا في الأسماء والـ formatting، بس بيضيف latency، وممكن "يصلّح" حاجات صح، فلازم تقيّده (يعدّل بس لما يكون فيه match صوتي معقول، وتقيس الـ WER قبل وبعد على test set).

اللي بيفرق: المرشح يعرف إن الـ biasing والـ LM fusion بيتظبطوا بـ hyperparameters (الـ LM weight والـ boost score) لازم تتعمل لها sweep على dev set من الـ domain، وإن الـ dev set ده لازم يكون فيه المصطلحات دي بنسبة واقعية، لأن الضبط على test set فيه المصطلحات في كل جملة بيبوظ الأداء العام.

**سؤال متابعة:** الـ LLM correction بيحوّل الأرقام المنطوقة "خمسة وعشرين ألف" لـ "25,000" وده ممتاز، بس في مرات بيخترع أرقام. ازاي تمنع ده؟

### س32. الـ language ID والـ dialect ID والـ keyword spotting: ازاي بيشتغلوا، وايه دورهم في voice system عربي؟

**الإجابة النموذجية:**
الـ spoken language ID بيصنّف الـ segment للغة، وغالبًا بيستخدم embeddings من موديل SSL أو من الـ STT encoder نفسه مع classifier بسيط، أو token الـ language في Whisper. الـ dialect ID أصعب بكتير لأن الفروق أدق والحدود مش حادة، والـ benchmarks العربية (ADI-5 وADI-17 من MGB-3 وMGB-5) بتبين إن الفرق بين اللهجات المتجاورة (النجدي والحجازي مثلًا) صعب حتى على البشر من segments قصيرة. الموديلات الحديثة بتعمل dialect ID كـ multi-task مع الـ STT أو بتاخد features من WavLM وبتدرب classifier عليها.

الاستخدام في النظام: (1) routing: توجيه لموديل STT أو voice مناسب، أو اختيار لهجة الرد في الـ TTS، وده مفيد لما المستخدمين متنوعين (عمالة من جنوب آسيا بتتكلم إنجليزي أو أردو، ومواطنين بلهجات مختلفة). (2) analytics: توزيع اللهجات في المكالمات. (3) evaluation slices: تقيس WER لكل لهجة. الخطأ الشائع إنك تعمل routing حاد على أساس dialect ID غير مؤكد فتوجّه لموديل أسوأ؛ الأفضل موديل STT واحد قوي على كل اللهجات، والـ dialect ID للـ analytics ولتكييف الـ TTS والـ LLM.

الـ keyword spotting (wake word زي "يا سلام") شغلانة مختلفة: موديل صغير جدًا (عشرات الآلاف من الـ parameters) شغّال دايمًا على الجهاز بـ compute واطي، بيتدرب على أمثلة إيجابية (الكلمة بأصوات ومسافات وضوضاء مختلفة، وغالبًا مع synthetic data من TTS) وسلبية كتير، والمقياس false accepts per hour مقابل false reject rate. بيتقيّم على ساعات طويلة من الكلام العادي والتلفزيون عشان الـ false accepts. للمنتجات الاستهلاكية العربية ده لازم يتبني محليًا لأن الكلمات العربية مش موجودة في الـ off-the-shelf systems.

**سؤال متابعة:** ليه لا ينفع تستخدم الـ STT الكبير كـ wake word detector؟ (compute دايم، latency، privacy، وdesign الـ false accept مختلف تمامًا).

### س33. الـ paralinguistics في المكالمات: الـ emotion recognition والـ audio events (ضحك، صمت، انتظار) والـ overlap detection. ايه اللي شغّال فعلًا وايه اللي hype؟

**الإجابة النموذجية:**
الـ speech emotion recognition من الصوت (مش النص) بيشتغل بـ features من موديلات SSL (WavLM أو emotion2vec) مع classifier، وبيتقاس على datasets ممثلة (IEMOCAP، MSP-Podcast). الحقيقة العملية: الدقة على الـ emotions الأساسية (غضب، حياد، فرح) معقولة لما الـ domain متطابق، بس بتنهار عبر الثقافات واللغات والقنوات: النبرة اللي بتعتبر غاضبة في dataset أمريكي مش هي في مكالمة خليجية، والتليفون بيقص أغلب الـ cues. وفيه مشكلة labels: الـ annotators نفسهم بيختلفوا. لو الـ use case "اكتشف العميل المستاء" غالبًا النص (بالـ LLM) زائد إشارات بسيطة من الصوت (ارتفاع الـ pitch والـ energy، سرعة الكلام، المقاطعات) بيدوا نتيجة أفضل من موديل عاطفة معقد.

الـ audio events المفيدة فعلًا في مراكز الاتصال: الصمت الطويل (العميل مستني)، موسيقى الانتظار، الضحك، الـ DTMF، الـ hold وتحويل المكالمة، ووجود متكلم تالت. دي بتتعمل بموديلات تصنيف بسيطة أو بقواعد على الـ VAD والـ energy وبتدخل في الـ QA metrics (نسبة الصمت، مين بيقاطع مين).

الـ overlap detection (اتنين بيتكلموا في نفس الوقت) مهم للـ diarization والـ STT وللـ analytics (مقاطعة الموظف للعميل)، وبيتعمل بموديل segmentation زي pyannote اللي بيطلّع classes: صمت، متكلم واحد، أكتر من متكلم. في التسجيلات الـ stereo (قناة لكل طرف) بيبقى مجرد مقارنة بين الـ VAD بتاع القناتين، وده سبب إضافي للإصرار على التسجيل الـ stereo في مراكز الاتصال.

**سؤال متابعة:** الـ product manager عايز "درجة رضا العميل من نبرة صوته" في dashboard. ازاي ترد؟

### س34. عندك 50 ألف ساعة data صوتي مجمّع من مصادر مختلفة بـ transcripts متفاوتة الجودة. ازاي تفلتره وتنضّفه قبل تدريب موديل STT؟

**الإجابة النموذجية:**
المبدأ: الـ label noise في الـ STT أخطر من الـ acoustic noise. الخطوات: (1) الـ metadata والـ dedup: إزالة التكرارات (نفس الملف بأسماء مختلفة، بالـ audio fingerprinting أو hash على الـ mel)، وإزالة أي حاجة بتتقاطع مع الـ test sets (contamination check على النص والمتكلم). (2) الـ language ID على الصوت والنص: نرمي اللي مش عربي أو اللي النص فيه لغة تانية غير الصوت. (3) الـ alignment-based filtering: نعمل forced alignment أو نحسب CTC loss per second بموديل مرجعي، والـ segments اللي الـ loss فيها عالي جدًا يا إما النص غلط يا الصوت مكسور. (4) الـ WER-based filtering: نشغّل موديل مرجعي (أو اتنين مختلفين) ونقارن بالـ transcript؛ لو الـ WER عالي جدًا نرمي أو نرسل للمراجعة، بس بحذر لأن ده بيرمي الحالات الصعبة (اللهجات) اللي الموديل المرجعي ضعيف فيها، فالـ threshold لازم يكون per-source أو per-dialect. (5) كشف الـ transcripts الآلية: Whisper في الـ paper عمل heuristics لاكتشاف الـ machine-generated subtitles (غياب علامات الترقيم، كل الحروف small، وهكذا) لأن تدريب الموديل على مخرجات موديل تاني بيورّث أخطاءه. (6) الـ acoustic filters: clipping، bandwidth، صمت طويل، موسيقى. (7) الـ text normalization الموحدة قبل التدريب، وإزالة الـ transcripts اللي فيها علامات annotators زي "[غير مفهوم]" أو التعامل معاها كـ tokens.

الأهم: تحتفظ بالإحصائيات لكل مصدر (كام ساعة راحت وليه)، وتدرب موديل صغير سريع على النسخة المفلترة مقابل غير المفلترة كـ ablation قبل التدريب الكبير، لأن الفلترة الزيادة ممكن تضر أكتر ما تنفع.

**سؤال متابعة:** الفلترة بالـ WER رمت 60% من data اللهجة الجنوبية و5% من MSA. ايه اللي حصل وازاي تتصرف؟

### س35. تقييم الـ STT بعيدًا عن الـ WER الإجمالي: entity accuracy، الأرقام، الـ slices، الـ latency metrics، والدلالة الإحصائية. ازاي تبني evaluation report يفيد في اتخاذ القرار؟

**الإجابة النموذجية:**
الـ WER الإجمالي رقم واحد بيخبي كل حاجة. الـ report المفيد فيه: (1) الـ slices: WER لكل لهجة، جنس المتكلم، قناة (تليفون، موبايل، استوديو)، مستوى الضوضاء، طول الـ utterance، ومجال الكلام. (2) الـ entity metrics: دقة الأرقام (كل رقم صح أو غلط، مش على مستوى الكلمة)، الأسماء، التواريخ، أرقام الهواتف والحسابات، لأن غلطة واحدة في رقم حساب أخطر من عشر أخطاء في كلمات ربط. (3) الـ error taxonomy: substitutions مقابل insertions مقابل deletions، وأكتر الكلمات اللي بتتغلط (confusion pairs)، عشان تعرف المشكلة acoustic ولا lexical ولا normalization. (4) الـ latency metrics للـ streaming: first partial latency، emission latency لكل كلمة (الفرق بين وقت نطقها ووقت ظهورها)، final latency بعد نهاية الكلام، وتوزيعهم p50 وp95 مش المتوسط. (5) الـ robustness: أداء تحت SNR مختلف وcodec مختلف. (6) الـ hallucination rate على مقاطع صمت وموسيقى. (7) الـ throughput والتكلفة.

الدلالة الإحصائية: الفرق بين 12.3% و11.9% ممكن يكون ضوضاء. تعمل paired bootstrap على مستوى الـ utterances (تعيد أخذ عينات وتحسب توزيع فرق الـ WER بين النظامين)، وتقرر حجم الـ test set على أساس الفرق اللي عايز تكتشفه (عشان تكتشف فرق 0.5% مطلق بثقة محتاج آلاف الجمل). ولازم الـ test set يكون مقفول (ما حدش يشوفه أثناء التطوير) وله dev set مقابل.

المرشح القوي بيختم بإن الـ report بيجاوب سؤال product: "نقدر نعتمد على النظام في X؟"، مش مجرد جدول أرقام، ولازم يكون فيه أمثلة حقيقية لأسوأ الأخطاء.

**سؤال متابعة:** الـ WER اتحسن 15% نسبيًا بس دقة أرقام الحسابات ساءت. ايه السيناريو اللي يعمل كده وأنهي تختار؟

### س36. الـ multi-talker ASR: الكلام المتداخل، الـ serialized output training، والـ target-speaker ASR. ايه الفرق بينهم وبين diarization ثم STT؟

**الإجابة النموذجية:**
في المحادثات الطبيعية 10-15% من الوقت فيه تداخل، والـ STT العادي بيتعامل مع الفترات دي كأنها متكلم واحد فبيطلّع كلام مشوّه أو بيختار واحد وبيرمي التاني. الـ pipeline التقليدي (diarization ثم STT لكل segment) ما بيحلش التداخل لأن الـ diarization نفسه بيكسر عند الـ overlap.

الـ SOT (serialized output training) بيدرب موديل AED واحد يطلّع كلام كل المتكلمين متسلسل مع token فاصل بين المتكلمين، مرتب حسب وقت البداية. الـ t-SOT (token-level) بيعمل نفس الفكرة على مستوى الـ tokens بحيث ينفع في الـ streaming بـ channel index لكل token. الموديلات دي بتتدرب على data مصنّع بتراكب utterances من متكلمين مختلفين، وأثبتت نفسها في الاجتماعات. الـ target-speaker ASR بياخد enrollment (كام ثانية من صوت المتكلم المستهدف) وبيطلّع كلامه هو بس، وده مفيد في الأجهزة الشخصية ومع الموظف في مركز الاتصال لما التسجيل mono.

الطريقة التانية: الـ speech separation أولًا (Conv-TasNet، SepFormer) ثم STT على كل مسار، بس الـ separation بتعمل artifacts وبتحتاج معرفة عدد المتكلمين، والـ continuous separation في الاجتماعات (CSS) بتحل جزء من ده.

عمليًا للسوق: في مراكز الاتصال الحل الأبسط والأقوى هو التسجيل بقناة لكل طرف، وفي الاجتماعات multi-talker ASR أو diarization متطور مع overlap-aware segmentation (pyannote بيعمل overlap detection وبيعيد تخصيص الـ overlap لمتكلمين). ولازم الـ evaluation يستخدم cpWER (concatenated minimum-permutation WER) اللي بيحسب الـ WER مع أحسن تطابق بين المتكلمين الحقيقيين والمكتشفين.

**سؤال متابعة:** ايه اللي بيحصل للـ SOT model لو دخل فيه 4 متكلمين وهو متدرب على 2 و3؟

### س37. الـ distillation والـ compression لموديلات STT: Distil-Whisper وWhisper turbo والـ pruning. ازاي تعمل موديل أسرع من غير خسارة كبيرة للعربي؟

**الإجابة النموذجية:**
في Whisper، الـ decoder (32 layer في large) هو اللي بيحدد الـ latency لأنه autoregressive: كل token بيمر على كل الـ layers. Distil-Whisper احتفظ بالـ encoder كامل ومجمّد، وعمل decoder بـ layer اتنين بس مُهيّأ من أول وآخر layer في المعلم، ودرّبه بـ KL divergence على توزيع المعلم زائد cross-entropy على pseudo-labels مفلترة (اللي الـ WER بينها وبين النص الأصلي أقل من threshold). النتيجة أسرع 5-6 مرات مع WER قريب، بس إنجليزي بس. Whisper large-v3-turbo عمل حاجة مشابهة لكن multilingual: 4 decoder layers مع fine-tuning على نفس data v3، والنتيجة أسرع بكتير مع فرق بسيط في الدقة، بس الـ translation task ضعف والعربي اللهجي ما اتحسنش.

للعربي لو عايز موديل أسرع: (1) تبدأ من turbo أو من موديل encoder-only (FastConformer CTC/TDT) اللي أصلًا أسرع 10 مرات من Whisper large. (2) لو لازم Whisper، تعمل distillation بنفسك: pseudo-labels من large-v3 (بعد fine-tuning على لهجتك) على آلاف الساعات غير معنونة، فلترة، وتدريب decoder صغير. (3) الـ pruning: structured pruning للـ attention heads أو الـ FFN dimensions ثم fine-tuning بيدي 20-40% تسريع، والـ unstructured pruning ما بيسرّعش على GPU عادي. (4) الـ quantization (int8 للـ weights والـ activations) على CTranslate2 أو TensorRT. (5) الـ speculative decoding: موديل صغير بيقترح tokens والكبير بيتحقق، وده بيسرّع الـ AED 2x من غير تغيير في الناتج.

القاعدة: تقيس الـ WER على test sets اللهجية بعد كل خطوة، لأن الـ compression بيضرب الحالات النادرة أولًا، وهي بالظبط اللهجات والأسماء.

**سؤال متابعة:** الـ distillation بالـ pseudo-labels بيورّث hallucinations المعلم. ازاي تقلل ده؟

---

## 3. الـ Text-to-Speech (TTS)

### س38. اشرح مراحل الـ TTS pipeline الكلاسيكي: text frontend، acoustic model، vocoder. وايه اللي VITS غيّره؟

**الإجابة النموذجية:**
**الـ frontend:** text normalization (أرقام، اختصارات، تواريخ، عملات)، ثم G2P أو phonemization (تحويل النص لـ phonemes)، وأحيانًا prosody annotations (pauses، stress). في العربي فيه خطوة إضافية قبل G2P: التشكيل التلقائي.

**الـ acoustic model:** بياخد الـ phoneme sequence ويطلع mel spectrogram. Tacotron 2: autoregressive مع attention، جودة ممتازة بس بطيء وبيعاني من attention failures (كلمات بتتكرر أو بتتشال). FastSpeech 2: non-autoregressive، بيستخدم variance adaptor بيتوقع duration وpitch وenergy لكل phoneme، ثم length regulator بيكرر كل phoneme حسب الـ duration، فبيبقى سريع وrobust، لكنه محتاج durations من MFA أو من teacher model وقت التدريب.

**الـ vocoder:** mel لـ waveform. WaveNet autoregressive جودة عالية لكن بطيء جدًا، WaveGlow (flows)، HiFi-GAN (GAN-based، سريع وجودة عالية، الـ default لسنين)، BigVGAN (تعميم أفضل على أصوات وأنواع صوت مختلفة)، Vocos (بيتوقع STFT coefficients ويعمل inverse STFT، أسرع).

**الـ VITS** دمج الـ acoustic model والـ vocoder في end-to-end model: conditional VAE مع normalizing flows وadversarial training، وMonotonic Alignment Search (MAS) بيتعلم الـ alignment بين النص والصوت أثناء التدريب من غير aligner خارجي، وstochastic duration predictor بيدي تنوع في الإيقاع. النتيجة: جودة أعلى وpipeline أبسط، ولسه VITS ومشتقاته (Piper مثلًا، وStyleTTS 2 بمنهج مختلف مبني على style diffusion) مستخدمين لما تحتاج speed وقلة موارد.

**سؤال متابعة:** ليه الموديلات الـ non-autoregressive كانت محتاجة duration predictor، وايه اللي حصل للفكرة دي في الـ flow-matching models؟

### س39. اشرح التحول لـ LLM-style TTS: الـ neural codec LMs (VALL-E)، CosyVoice، الـ flow matching (F5-TTS، E2 TTS)، وOrpheus. الـ tradeoffs بينهم.

**الإجابة النموذجية:**
**موديلات الـ neural codec LMs (VALL-E):** بتعامل الـ TTS كـ language modeling على acoustic tokens من codec (EnCodec RVQ بـ 8 codebooks). AR model بيتوقع الـ codebook الأول، وNAR model بيتوقع الباقي. أول مرة نشوف zero-shot voice cloning حقيقي من 3 ثواني prompt. المشاكل: instability (كلمات بتتكرر أو بتتشال، WER أعلى من الـ classic models)، بطء الـ AR، وصعوبة التحكم.

**موديل CosyVoice (Alibaba):** LLM بيتوقع supervised semantic tokens (من ASR encoder مع quantizer، فالـ tokens مربوطة بالمحتوى مش بالـ acoustics)، ثم flow-matching model بيحوّلها لـ mel مع speaker embedding، ثم vocoder. الفصل ده بين المحتوى والصوت بيدي stability أعلى وcloning كويس. CosyVoice 2 أضاف streaming بـ chunk-aware causal flow matching وتحكم بالتعليمات، وCosyVoice 3 حسّن الـ multilingual والـ data scale.

**موديلات الـ flow matching غير الـ AR (E2 TTS، F5-TTS، Voicebox):** diffusion transformer بيتعلم يحوّل noise لـ mel مباشرة، والنص بيتحط كـ characters مع filler tokens لحد طول الـ audio، من غير phonemes ولا duration model ولا alignment صريح. سريع جدًا (كام NFE steps)، وجودة عالية، وcloning ممتاز، لكن: محتاج تقدير للطول الكلي (F5 بيقدره من نسبة النص للـ prompt)، وأحيانًا بيسقط أو يكرر كلمات في الجمل الطويلة، ومش streaming بطبيعته.

**موديلات زي Orpheus (Llama backbone + SNAC codec):** LLM نصي جاهز بيتعمله fine-tuning ليطلع audio tokens مباشرة، فبيستفيد من فهم اللغة والـ streaming الطبيعي للـ AR وinfrastructure الـ LLM serving (vLLM). وعلى الجهة التانية Kokoro: موديل صغير جدًا مبني على StyleTTS 2 بجودة كويسة وسرعة عالية على CPU لكن من غير cloning.

**الـ tradeoffs:** الـ AR بيدي streaming طبيعي وprosody أطول نفسًا لكن أبطأ وأقل استقرار. الـ NAR سريع وrobust لكن محتاج chunking للـ streaming. الـ LLM-based بيدي تحكم بالتعليمات والعاطفة. الـ classic (VITS) أرخص وأكثر deterministic. للعربي: أغلب الموديلات دي متدربة على English/Chinese بشكل أساسي، والـ fine-tuning محتاج data عربية مشكّلة أو على الأقل consistent، وموضوع الـ license مهم (هنرجعله في قسم الـ serving).

**سؤال متابعة:** ليه الـ semantic tokens في CosyVoice بتقلل الـ hallucination مقارنة بالـ acoustic tokens في VALL-E؟

### س40. الـ neural audio codecs: EnCodec وSoundStream وDAC وSNAC وMimi. اشرح RVQ، والفرق بين semantic وacoustic tokens، وليه الـ frame rate مهم.

**الإجابة النموذجية:**
الـ neural codec: encoder بيحوّل الـ waveform لـ latents بـ frame rate واطي، quantizer بيحوّلها لـ discrete tokens، وdecoder بيرجّع الـ waveform. بيتدرب بـ reconstruction loss (multi-scale spectral) + adversarial loss (discriminators) + commitment loss للـ codebooks، وأحيانًا perceptual loss.

**الـ RVQ (Residual Vector Quantization):** بدل codebook واحد ضخم، بتستخدم سلسلة codebooks: الأول بيقرّب الـ latent، والتاني بيقرّب الـ residual، وهكذا. الـ codebook الأول بيشيل المعلومة الأهم (content تقريبًا) والباقي تفاصيل (timbre، noise). الـ bitrate = frame rate × عدد الـ codebooks × log2(حجم الـ codebook). مثلًا EnCodec عند 75 Hz بـ 8 codebooks من 1024 = 6 kbps. SNAC بيستخدم codebooks بـ frame rates مختلفة (multi-scale). Mimi (Kyutai) بيشتغل عند 12.5 Hz فقط وبيعمل distillation لـ WavLM في الـ codebook الأول عشان يكون semantic.

**الفرق بين الـ semantic والـ acoustic tokens:** الـ semantic tokens (HuBERT k-means، أو الـ supervised tokens في CosyVoice) بتمثل المحتوى اللغوي وبتتجاهل الـ speaker والـ acoustics، فسهل على الـ LM يتعلم عليها لكن مش كافية لإعادة بناء الصوت. الـ acoustic tokens بتحتفظ بكل التفاصيل لكن الـ sequence طويلة ومليانة معلومة مش لغوية. الحل الشائع: hierarchical (semantic أولًا ثم acoustic)، أو codec بيدمج الاتنين (SpeechTokenizer، Mimi، X-codec).

**ليه الـ frame rate مهم:** عند 75 Hz دقيقة واحدة = 4500 token لكل codebook، وده بيقتل الـ LLM context والـ speed. الاتجاه الحالي: 12.5 إلى 25 Hz مع single codebook أو codebooks قليلة (WavTokenizer، BigCodec)، عشان الـ speech LM يشوف sequence قريبة من النص.

**سؤال متابعة:** ايه هي الـ codebook collapse وازاي بتتحل؟ (EMA updates، dead code re-initialization، factorized codes وL2-normalized lookup زي DAC).

### س41. الـ Arabic TTS: التشكيل وG2P والـ text normalization واللهجات. ازاي تبني الـ frontend؟

**الإجابة النموذجية:**
المشكلة الجوهرية إن الكتابة العربية بتحذف الحركات القصيرة، فـ "كتب" ممكن تكون كَتَبَ أو كُتِبَ أو كُتُب، والنطق مختلف تمامًا. عندك ثلاث استراتيجيات:

1. **Diacritizer ثم rule-based G2P:** موديل تشكيل تلقائي (CATT، Shakkala، fine-tuned BERT/ByT5، أو LLM prompting بجودة أعلى لكن أبطأ وأغلى) يشكّل النص، ثم قواعد نطق الفصحى وهي منتظمة جدًا بعد التشكيل: الشدة، التنوين، همزة الوصل، إدغام لام التعريف مع الحروف الشمسية، التاء المربوطة عند الوقف، المد، الألف الخنجرية في كلمات زي "هذا" و"لكن" و"الرحمن". الـ pipeline ده الأكثر تحكمًا، ودقة التشكيل هي bottleneck الجودة.
2. **End-to-end على نص مشكّل:** تدرب الموديل على characters بالتشكيل، وتعمل التشكيل وقت الـ inference. أبسط لكن لسه معتمد على الـ diacritizer.
3. **End-to-end على نص غير مشكّل:** الموديلات الكبيرة (LLM-based TTS) بتتعلم النطق من السياق، وبتنجح في الكلام الشائع، بس بتغلط في الكلمات الغامضة والأسماء، ومفيش طريقة تصلّح غلطة غير بالتشكيل اليدوي. عمليًا الأفضل: تدريب على mix، وقبول تشكيل اختياري من المستخدم كـ override.

**الـ text normalization** في العربي أصعب من الإنجليزي: الأرقام لازم تتقرأ بجنس المعدود وحالته (ثلاثة كتب، ثلاث سيارات)، التواريخ الهجرية والميلادية، العملات (ريال/هللة)، النسب، أرقام التليفونات تتقرأ رقم رقم، الاختصارات، والـ code-switching لكلمات إنجليزية (تتنطق إنجليزي ولا تتعرّب؟)، وأسماء أجنبية محتاجة transliteration.

**اللهجات:** مفيش orthography ولا diacritizer قوي للهجات، فالحل العملي: اتفق على conventions كتابة ثابتة في الـ dataset، وشكّل الـ training data بالكامل بواسطة لغويين، وخلي الموديل يتعلم اللهجة من الـ data نفسها مش من قواعد.

**التقييم:** مستمعين native من نفس اللهجة، وقائمة اختبار فيها الأرقام والأسماء والكلمات الغامضة والجمل الطويلة، وpronunciation error rate يحسبه لغوي.

**سؤال متابعة:** ازاي تتعامل مع كلمة زي "مصر" في الفصحى مقابل المصري (مِصْر مقابل مَصْر)، وايه الـ override mechanism اللي تديه للـ product team؟

### س42. الـ voice cloning: zero-shot مقابل fine-tuning، speaker embeddings مقابل in-context prompting، الجودة والـ safety.

**الإجابة النموذجية:**
**الـ fine-tuning لكل متكلم:** بتاخد من 10 دقايق لساعات وتعمل fine-tuning للموديل (أو adapter/LoRA). أعلى جودة وثبات، خصوصًا للـ brand voice بتاعة شركة، لكن مكلف ومحتاج pipeline لكل صوت.

**الـ zero-shot بـ speaker encoder:** موديل speaker verification (ECAPA-TDNN، أو encoder متدرب مع الـ TTS زي YourTTS) بيطلع embedding ثابت من ثواني، والـ TTS بيتشرط عليه. رخيص لكن الـ similarity محدودة، وبيضيّع تفاصيل الأسلوب.

**الـ in-context prompting:** الموديل بياخد الـ audio prompt وtranscript بتاعه كجزء من الـ input (VALL-E، XTTS، F5-TTS، CosyVoice) وبيكمل بنفس الصوت والأسلوب. أعلى similarity في zero-shot وبينقل الـ prosody والـ accent، لكن حساس لجودة الـ prompt (noise، طول 3 إلى 10 ثواني، نهاية الجملة)، وممكن ينقل accent اللغة الأصلية في الـ cross-lingual cloning (متكلم إنجليزي يطلع عربي بلكنة).

**التقييم:** speaker similarity بـ cosine على embeddings (WavLM-based ECAPA أو Resemblyzer)، naturalness (MOS/UTMOS)، intelligibility (WER بالـ ASR)، وقياس accent leakage بالسمع.

**الـ safety والامتثال:** التحقق من موافقة صاحب الصوت (consent recording أو عقد)، منع cloning أصوات عامة، audio watermarking (زي AudioSeal) على المخرجات، deepfake detection كطبقة دفاع، وتسجيل من استخدم الخدمة (audit log). في السعودية موضوع الـ PDPL والبيانات البيومترية مهم، والصوت بيعتبر بيانات شخصية حساسة.

**سؤال متابعة:** لو الـ brand voice لبنك لازم يكون ثابت 100% في كل مكالمة، بتختار أنهي approach وليه؟ (fine-tuned voice مع sampling ثابت وtemperature واطية، مش zero-shot).

### س43. الـ streaming TTS: ازاي توصل لـ time-to-first-audio واطي؟ الـ chunking والـ AR مقابل NAR والحدود بين الـ chunks.

**الإجابة النموذجية:**
في voice agent الـ TTS مش بيبدأ لما الـ LLM يخلص، لكن مع أول جملة. الـ pipeline: الـ LLM بيطلع tokens streaming، ثم sentence/clause splitter بيجمّع لحد علامة وقف طبيعية (نقطة، فاصلة، أو حد أقصى من الكلمات)، ثم يبعت الـ chunk للـ TTS، والـ TTS يطلع audio streaming للـ client.

**الموديلات الـ AR (codec LMs، Orpheus، CosyVoice 2):** بتطلع audio tokens واحدة ورا التانية وبتتحول لـ audio كل كام frame، فالـ TTFB ممكن يكون أقل من 200 ms، وCosyVoice 2 حل مشكلة الـ flow-matching stage بـ chunk-aware causal design مع lookahead محدود.

**الموديلات الـ NAR (F5-TTS، VITS):** لازم تولّد الـ utterance كلها، فالـ streaming بيبقى على مستوى الـ text chunks بس، وكل chunk بيتولد كامل. الحل: chunks قصيرة في البداية (أول جملة أو أول 8 كلمات) وأطول بعد كده، مع الحفاظ على معنى الجملة.

**المشاكل عند الحدود:** prosody discontinuity (الجملة بتبان مقطوعة، الـ pitch بيقفز)، فالحلول: crossfade قصير بين الـ chunks، تمرير آخر ثانية من الـ audio السابق كـ prompt (context continuation)، وتجنب الـ chunks الأقصر من كام كلمة. كمان الـ sentence splitter لازم يفهم الاختصارات والأرقام العشرية (25.5 مش نهاية جملة).

**القياسات:** TTFB للـ TTS (الهدف 100 إلى 300 ms)، RTF لكل chunk لازم يكون أقل من 1 بمسافة عشان الـ playback ميقفش، وقياس الـ underruns في الـ client buffer.

**سؤال متابعة:** لو الـ LLM بيطلع tokens أبطأ من ما الـ TTS بينطقهم، ايه اللي يحصل وازاي تتعامل؟ (الـ buffer بيفضى، فتحتاج تجمّع جمل أطول قبل البداية، أو LLM أسرع، أو filler طبيعي).

### س44. تقييم الـ TTS: MOS وCMOS وMUSHRA والـ automatic metrics والـ WER بالـ ASR والـ speaker similarity. وايه الـ pitfalls؟

**الإجابة النموذجية:**
**التقييم الـ subjective:** MOS على مقياس 1 إلى 5 هو الأشهر، لكنه absolute وبيتأثر بالمقيّمين والسياق، فمش صالح للمقارنة بين أوراق أو تجارب مختلفة، ومحتاج عينات كتير (على الأقل 20 مقيّم لكل عينة) عشان الـ confidence interval يبقى مفيد. CMOS (comparison MOS) بيقارن نظامين على نفس الجملة من -3 إلى +3، أكثر حساسية وأنسب لقرارات "الموديل الجديد أحسن من القديم ولا لأ". MUSHRA بيعرض عدة أنظمة مع hidden reference وanchor، وأدق لكن أغلى. لازم المقيّمين يكونوا native في نفس اللهجة، والجمل تكون من نفس الـ domain.

**التقييم الـ objective/automatic:** UTMOS وNISQA بيتوقعوا MOS، مفيدين للمراقبة المستمرة والـ ablations مش للقرارات النهائية. DNSMOS للجودة والضوضاء. الـ intelligibility: تعمل ASR للصوت المولّد وتحسب WER/CER مقابل النص الأصلي، بموديل ASR قوي في العربي وبنفس الـ normalization، مع إدراك إن الـ ASR بيكافئ الكلام المبالغ في وضوحه. الـ speaker similarity بالـ cosine على speaker embeddings. متريكس أقدم زي MCD وF0 RMSE بتحتاج ground truth بنفس الـ alignment ومش بتتستخدم كتير دلوقتي.

**مقاييس الـ production:** RTF، TTFB، معدل الأخطاء في النطق على قائمة اختبار ثابتة (أرقام، أسماء، code-switching)، ومعدل الـ failures (صمت، تكرار، ضوضاء).

**الـ pitfalls:** MOS بيتشبع فوق 4.3 فمش هيفرّق بين موديلين كويسين، الـ rater fatigue، مقارنة أرقام MOS من ورقتين مختلفتين، الاعتماد على WER لوحده (موديل ممكن يكون واضح وممل)، وتجاهل الـ long-form (الموديل ممكن يكون ممتاز في جملة وسيء في فقرة).

**سؤال متابعة:** ازاي تصمم A/B test لصوتين في IVR حقيقي وتقيس أثره على containment rate بدل الـ MOS؟

### س45. الـ prosody والتعبير: ازاي الموديلات بتتحكم في العاطفة والسرعة والتشديد؟ SSML وstyle tokens وinstruction-based TTS.

**الإجابة النموذجية:**
الـ prosody = pitch وduration وenergy وpauses. طرق التحكم:

- **التحكم المباشر في الـ variance:** في FastSpeech 2 وأشباهه تقدر تضرب الـ predicted duration أو pitch في factor، فتتحكم في السرعة والنغمة بشكل deterministic.
- **الـ style/reference encoders:** GST (Global Style Tokens) وreference encoders بتستخرج style embedding من audio مرجعي، فتقدر تقول "اتكلم زي المقطع ده".
- **الـ SSML:** في المحركات التجارية: prosody rate/pitch/volume، break time، emphasis، say-as لقراءة الأرقام والتواريخ. مفيد جدًا في IVR لأنه deterministic، لكن الموديلات الحديثة الـ open-source مش دايمًا بتدعمه.
- **التحكم بالتعليمات النصية:** CosyVoice-Instruct وParler-TTS بياخدوا وصف نصي ("صوت هادئ وبطيء")، وبعض المحركات التجارية بتستخدم audio tags داخل النص، والـ LLM-based TTS بيتعلم يفهم علامات الترقيم والسياق (سؤال، تعجب).
- **الـ sampling parameters:** temperature وCFG scale في الـ flow matching بتأثر في التنوع مقابل الثبات.

في الـ voice agents المطلوب غالبًا ثبات مش تعبير مبالغ فيه، ولازم الـ style يبقى ثابت عبر الـ chunks. للعربي: الـ expressive data شحيحة، والتشديد في العربي (stress) مرتبط بالوزن الصرفي، فالموديل محتاج data كافية يتعلم منها، ومفيش أدوات annotation جاهزة.

**سؤال متابعة:** ازاي تخلي الصوت يقرأ رقم حساب ببطء ووضوح من غير ما يبقى بطيء في باقي الجملة؟

### س46. الـ flow matching مقابل الـ diffusion مقابل الـ GAN في توليد الصوت: اشرح الفكرة الرياضية باختصار، والـ NFE والـ CFG، وليه الـ flow matching بقى الـ default في الـ TTS.

**الإجابة النموذجية:**
الثلاثة بيحلوا نفس المشكلة: توليد mel أو latent أو waveform من توزيع معقد. الـ GAN بيتعلم generator بخطوة واحدة عن طريق discriminator بيفرّق الحقيقي من المولّد؛ سريع جدًا في الـ inference (خطوة واحدة) بس التدريب غير مستقر وفيه mode collapse، وعشان كده بقى محصور في الـ vocoders (HiFi-GAN) والـ end-to-end models زي VITS. الـ diffusion (DDPM) بيتعلم يشيل ضوضاء تدريجيًا: بتضيف noise على data بجدول معين، والشبكة بتتعلم تتوقع الـ noise (أو الـ score)، والـ sampling بيحتاج مئات أو آلاف الخطوات (أو عشرات مع DDIM)، وده بطيء للـ TTS.

الـ flow matching بيتعلم vector field بيحرّك الـ samples من الـ noise للـ data على مسار مستقيم (optimal transport path): بتاخد x0 من الـ Gaussian وx1 من الـ data، وتعمل interpolation x_t = (1 - t) x0 + t x1، وتدرب الشبكة تتوقع الاتجاه x1 - x0 عند x_t وt. الـ loss regression بسيط ومستقر جدًا، والمسارات المستقيمة معناها إنك تقدر تحل الـ ODE بخطوات قليلة (Euler أو midpoint بـ 8-32 خطوة) بجودة عالية. ده اللي خلى Voicebox وMatcha-TTS وF5-TTS وdecoder CosyVoice 2 كلهم flow matching.

الـ NFE (number of function evaluations) هو عدد مرات تشغيل الشبكة في الـ sampling، وهو الـ knob الأساسي للـ latency مقابل الجودة؛ F5 بيشتغل كويس عند 16-32 وبيبدأ يتكسر تحت 8، والـ distillation (rectified flow أو consistency models) بيوصّل لخطوة أو اتنين. الـ CFG (classifier-free guidance) بيدرب الشبكة أحيانًا من غير الشرط (النص) وفي الـ inference بيحسب الاتجاه مرتين (بشرط ومن غير) ويكبّر الفرق بينهم بمعامل w (حوالي 2 في F5)؛ بيحسّن الوضوح والالتزام بالنص على حساب ضعف التكلفة وتقليل التنوع، ولو زودته بيطلع صوت متكلّف.

المرشح القوي بيذكر إن الـ flow matching بيولّد mel أو latent مش waveform، فلسه محتاج vocoder، وإن الـ latent diffusion على codec latents (NaturalSpeech 2 وStable Audio) بديل مهم.

**سؤال متابعة:** ليه الـ AR codec LM لسه موجود مع إن الـ flow matching أسرع وأحسن جودة في حالات كتير؟ (الـ streaming الطبيعي، والقدرة على النمذجة اللغوية للـ prosody الطويلة، وسهولة الدمج مع LLM).

### س47. الـ duration modeling والـ alignment في الـ TTS: الـ attention-based (Tacotron)، والـ MAS (Glow-TTS وVITS)، والـ external alignment (FastSpeech)، والـ alignment-free (VALL-E وF5). وايه failure modes كل طريقة؟

**الإجابة النموذجية:**
الـ TTS محتاج يعرف كل حرف/phoneme بيقابل كام frame. Tacotron كان بيتعلم ده بـ attention مرن، والنتيجة أخطاء مشهورة: تكرار كلمات، تخطي كلمات، وعدم التوقف في نهاية الجملة، خصوصًا في الجمل الطويلة أو النادرة. FastSpeech حل ده بـ duration predictor صريح متدرب على durations من forced alignment (MFA) أو من attention معلّم، فبقى الـ output متوازي وسريع ومن غير تكرار، لكن جودة الـ prosody بقت محدودة بجودة الـ durations وبالـ regression للمتوسط.

Glow-TTS قدّم الـ MAS (monotonic alignment search): خوارزمية Viterbi بتلاقي الـ alignment الأحادي الاتجاه اللي بيعظّم الـ likelihood تحت الموديل نفسه، من غير aligner خارجي، والـ VITS استخدمها مع stochastic duration predictor مبني على flows عشان يعمل تنوع في الإيقاع. المشكلة إن الـ MAS بيحتاج الموديل يكون likelihood-based (flow أو VAE)، وبيعمل alignments غلط في بداية التدريب أحيانًا.

الموديلات الـ LLM-style (VALL-E وCosyVoice) alignment-free: الـ decoder بيولّد audio tokens لحد ما يطلّع EOS، والـ attention بتتعلم الـ alignment ضمنيًا زي الترجمة. الـ failure modes رجعت: توقف مبكر قبل نهاية النص، استمرار بعد النهاية (run-on)، تكرار، تخطي، وقراءة غلط للأرقام والاختصارات. الـ flow matching non-AR models زي F5 بتحتاج الطول الكلي مقدمًا (بتقدّره من نسبة طول النص للطول في الـ prompt)، ولو التقدير غلط الكلام بيتسرّع أو بيتمطّ أو بتتقص الجملة.

في الـ production، الحل العملي: ASR على الناتج ومقارنة بالنص (word-level)، وإعادة التوليد بـ seed مختلف لو فيه فرق، وقياس نسبة إعادة التوليد كـ metric. والـ duration control الصريح (زي اللي في Index-TTS 2 ونظم الـ dubbing) بيرجع مهم لما محتاج الكلام يطلع في وقت محدد.

**سؤال متابعة:** الـ MAS بيعمل alignment أحادي الاتجاه صارم. ايه الحالات اللغوية اللي ده بيبقى فيها تقريب مش دقيق؟ (الـ coarticulation والحروف اللي بتتنطق مع بعض، والحذف في الكلام السريع).

### س48. الـ vocoders بعمق: اشرح HiFi-GAN (الـ generator والـ discriminators والـ losses)، وايه اللي BigVGAN وVocos وHiFT ضافوه، وليه الـ vocoder بيتكسر لما يشوف mel من acoustic model مختلف.

**الإجابة النموذجية:**
HiFi-GAN generator بياخد mel وبيعمل upsampling بـ transposed convolutions على مراحل (مثلًا 8x 8x 2x 2x عشان يوصل من hop 256 للـ waveform)، وبعد كل مرحلة multi-receptive field fusion: مجموعة residual blocks بـ kernel sizes وdilations مختلفة بتتجمع، عشان يلقط patterns بمقاييس زمنية مختلفة. الـ discriminators اتنين: الـ MPD بيعيد تشكيل الـ waveform لمصفوفة 2D بفترات (2 و3 و5 و7 و11) عشان يلقط الـ periodic structure بتاع الـ harmonics، والـ MSD بيشتغل على الـ waveform بثلاث scales (raw ثم avg-pooled مرتين). الـ losses: adversarial (least squares)، وfeature matching (المسافة بين الـ intermediate features في الـ discriminator للحقيقي والمولّد)، وmel-spectrogram L1 بوزن كبير (حوالي 45) عشان يثبّت التدريب.

BigVGAN غيّر الـ activation لـ snake (x زائد مركّبة جيبية) بيدي inductive bias للدورية، وأضاف anti-aliasing (low-pass filtering حوالين الـ up/down sampling) عشان الـ activations غير الخطية ما تعملش aliasing، ودرّب على بيانات ضخمة ومتنوعة، فبقى أقوى بكتير على أصوات ومتكلمين مش موجودين في التدريب. Vocos بدل ما يعمل upsampling تدريجي، بيتوقع الـ STFT (magnitude وphase) مباشرة بـ ConvNeXt backbone ويعمل iSTFT، فبقى أسرع 10 مرات وبيشتغل كويس على CPU. HiFT (في CosyVoice) بيجمع HiFi-GAN مع iSTFT وneural source filter مشروط بالـ F0 عشان يثبّت الـ pitch.

الـ mismatch: الـ vocoder اتدرب على mel حقيقي، والـ acoustic model بيطلّع mel أنعم وأقل تفصيلًا (over-smoothed) أو بـ distribution شوية مختلفة، فالـ vocoder بيطلّع صوت مكتوم أو فيه buzz. الحلول: fine-tuning للـ vocoder على الـ mels اللي الـ acoustic model بيطلّعها فعلًا لنفس الـ training data (ground-truth aligned fine-tuning)، أو end-to-end training زي VITS، أو vocoder قوي عام زي BigVGAN مع أخذ الـ mel configuration بالمليمتر. وسبب تاني شائع: اختلاف mel parameters (fmax، log base، normalization) بين الاتنين.

**سؤال متابعة:** ليه الـ vocoders بتتعب في الـ high frequencies والـ breath وأصوات s وsh بالذات، وايه اللي بيساعد؟ (الـ phase غير متوقعة في الـ noise-like sounds، والـ MPD والـ multi-resolution STFT loss بيساعدوا).

### س49. تصميم موديل TTS واحد متعدد المتكلمين واللهجات: الـ conditioning، والـ data balancing، ومشكلة الـ entanglement بين المتكلم واللهجة. وامتى تفضّل موديل لكل صوت؟

**الإجابة النموذجية:**
الـ conditioning بيتعمل بـ speaker embedding (lookup table للمتكلمين المعروفين، أو x-vector/ECAPA/reference encoder للـ zero-shot) وdialect/style token أو prompt. المزايا: مشاركة المعرفة اللغوية والصوتية بين الأصوات (الصوت اللي عنده ساعتين بيستفيد من اللي عنده عشرين)، وموديل واحد في الـ serving، وإمكانية خلط الخصائص. المشاكل: (1) الأصوات ذات الجودة الأقل بتسحب الجودة العامة، فلازم فلترة وربما وزن أقل. (2) الـ imbalance: صوت بـ 50 ساعة وصوت بـ ساعة، فبتعمل sampling بيرفع الأصوات الصغيرة بحدود. (3) الـ entanglement: لو كل متكلم بيتكلم بلهجة واحدة، الموديل بيتعلم إن اللهجة جزء من هوية الصوت، فلما تطلب من الصوت النجدي يتكلم مصري بيغيّر الصوت نفسه أو ما بيستجيبش. الحل إنك تجيب متكلمين بيتكلموا أكتر من لهجة، أو تستخدم adversarial classifier يمنع الـ speaker embedding من حمل معلومات اللهجة، أو تقبل الواقع وتعتبر (صوت + لهجة) كيان واحد.

موديل لكل صوت (single-speaker fine-tune) لسه أحسن جودة للـ brand voice لما عندك 10 ساعات نظيفة من الصوت ده، لأن الموديل بيتخصص وما فيش تسريب، وسهل تحديثه وترقيمه. الـ multi-speaker أحسن لما عندك أصوات كتير قليلة الـ data أو محتاج zero-shot.

عمليًا للسوق السعودي: موديل multi-speaker/multi-dialect كـ base، ثم fine-tune لكل brand voice منه، وتثبّت voice version لأن العميل بيلاحظ أي تغير في "صوت الشركة".

**سؤال متابعة:** ازاي تتحقق إن الـ speaker embedding ما بيحملش معلومات عن النص أو اللهجة؟ (تدريب probe classifier على الـ embeddings).

### س50. الـ long-form TTS (كتب صوتية، مقالات، تقارير): ايه المشاكل اللي بتظهر لما تولّد نص طويل، وازاي تحافظ على الاتساق في الصوت والـ prosody والـ pauses؟

**الإجابة النموذجية:**
الموديلات بتتدرب على جمل قصيرة (2-20 ثانية)، فلما تدّيها فقرة طويلة الجودة بتتدهور: الـ AR models بتنسى أو بتكرر، والـ NAR بتحتاج تقدير طول غلط. فالحل الأساسي تقطيع النص لجمل أو جمل قصيرة، وتوليد كل chunk، ثم الدمج. المشاكل اللي بتظهر: (1) عدم اتساق الصوت بين الـ chunks (في الـ zero-shot models كل chunk ممكن يطلع بنبرة أو حتى بطابع صوتي شوية مختلف). (2) الـ prosody: كل جملة بتبدأ كأنها أول جملة، فالفقرة بتبان مقطّعة. (3) الـ pauses: الوقفة بين الجمل والفقرات والعناوين لازم تتحكم فيها صراحة. (4) الـ loudness والـ pitch level بيتغيروا. (5) النطق مش متسق: نفس الاسم بيتنطق بطريقتين في الفصل نفسه.

الحلول: تثبيت الـ speaker prompt نفسه (ونفس الـ seed للموديلات الـ stochastic)، واستخدام موديلات بتدعم context continuation (تدي الموديل آخر كام ثانية من الـ chunk السابق كـ prompt صوتي زائد نصه، وده بيخلي الانتقال طبيعي)، وتحكم صريح في الـ pauses بالـ SSML أو بإدراج صمت محسوب حسب نوع الفاصل، وloudness normalization على الناتج الكامل، وlexicon موحد للأسماء، وpre-processing للنص (عناوين، قوائم، جداول، اقتباسات، أرقام صفحات) لأن النص المكتوب مش مصمم للقراءة بصوت عالي. ولإبراز النص وقت التشغيل (highlighting) بتعمل forced alignment على الناتج لأن timestamps الموديل غير موثوقة.

المقياس: تقييم بشري على فقرات كاملة مش جمل، وWER بالـ ASR على الكتاب كله لاكتشاف الجمل المتخطاة، وعدد مرات إعادة التوليد. الكتب الصوتية التجارية لسه بتحتاج مراجعة بشرية على الأقل عينات، وده جزء من التكلفة.

**سؤال متابعة:** الفصل فيه حوار بين شخصيات. ازاي تتعامل معاه من غير ما تبني نظام dubbing كامل؟

### س51. الـ cross-lingual والـ code-switched TTS: صوت عربي بيقول مصطلحات وأسماء إنجليزية. ليه بيطلع بـ accent غريب أو بيقرا الكلمة الإنجليزية غلط، وايه الحلول؟

**الإجابة النموذجية:**
المشكلة الأساسية entanglement: الموديل شاف الصوت ده بيتكلم عربي بس، فتعلّم إن الخصائص الصوتية والنطق الإنجليزي حاجة واحدة. لما تطلب منه إنجليزي، يا إما يطلّع إنجليزي بلكنة عربية ثقيلة (وده أحيانًا مطلوب لأنه طبيعي للمتكلم)، يا إما يبدّل الصوت لصوت متكلم إنجليزي شافه في التدريب، يا إما يتعامل مع الحروف اللاتينية كأنها رموز ويخترع نطق. والـ frontend نفسه مشكلة: الـ G2P العربي ما يعرفش يقرا "Wi-Fi".

الحلول على مستويات: (1) في الـ frontend: تحويل الكلمات الإنجليزية الشائعة لكتابة عربية صوتية قبل الموديل ("واي فاي") بقاموس تحويل، ودي أبسط طريقة وبتدي نطق متسق بلكنة محلية طبيعية، بس ما بتنفعش للأسماء الجديدة كل يوم. (2) phoneme set موحد (IPA أو مشابه) مع language embedding لكل جزء من النص، فالموديل بيشوف phonemes مش حروف، وبيتعلم إن نفس الـ phoneme بيتنطق بنفس الطريقة بغض النظر عن اللغة. (3) data: تسجيل جمل مختلطة من نفس المتكلم (وده أهم حاجة للـ brand voice)، أو توليد data مختلط بـ voice conversion. (4) الموديلات الـ LLM-style الـ multilingual (XTTS وCosyVoice وأمثالها) بتتعامل مع الـ switching ضمنيًا لأن الـ tokenizer نصي عام، بس درجة الـ accent بتعتمد على الـ speaker prompt.

الاختيار بين "لكنة محلية" و"نطق native" قرار product مش تقني: في مركز اتصال سعودي، "OK" و"iPhone" بلكنة سعودية خفيفة أطبع من نطق أمريكي فجائي في نص الجملة.

**سؤال متابعة:** ازاي تقيّم جودة نطق الكلمات الإنجليزية جوه الجمل العربية بشكل كمي؟ (ASR إنجليزي على الأجزاء المقطوعة، أو phoneme recognition ومقارنة بالنطق المرجعي، زائد تقييم بشري).

### س52. الـ voice conversion: ازاي بيشتغل (kNN-VC وRVC والـ disentanglement)، وايه استخداماته في الـ data pipeline والـ privacy، وايه مخاطره؟

**الإجابة النموذجية:**
الـ voice conversion بيغيّر هوية المتكلم مع الحفاظ على المحتوى والـ prosody (غالبًا). الفكرة الحديثة: تستخرج content representation قليلة المعلومات عن المتكلم (features من layers وسطى في HuBERT أو WavLM، أو الـ discrete semantic tokens)، وتضيف ليها الهوية المستهدفة، وتعيد بناء الصوت. kNN-VC بسيط لدرجة مدهشة: بياخد WavLM features للمصدر، وبيبدّل كل frame بأقرب frames ليها في features المتكلم الهدف (matching set)، ويمرر الناتج على vocoder؛ من غير تدريب لكل متكلم. RVC (اللي انتشر في الـ community) بيستخدم HuBERT content زائد pitch extraction زائد VITS decoder متدرب لكل متكلم مع retrieval من features المتكلم. الموديلات الأحدث (Seed-VC وأمثالها) diffusion/flow-based وzero-shot.

الاستخدامات المشروعة: (1) data augmentation للـ STT: تحويل نفس الـ utterances لأصوات مختلفة عشان تزيد تنوع المتكلمين في اللهجات النادرة. (2) الـ anonymization: مشاركة تسجيلات مكالمات للبحث أو للـ vendors بعد تغيير الصوت (VoicePrivacy challenge بتقيّم ده)، مع ملاحظة إن المحتوى نفسه ممكن يكون PII. (3) إنتاج data لـ TTS: تحويل تسجيلات استوديو لصوت الـ brand voice لزيادة ساعاته (بحذر شديد لأن الـ artifacts بتتعلم). (4) الـ dubbing.

المخاطر: انتحال الشخصية (الـ VC أسهل من الـ TTS cloning لأنه محتاج بس صوت الهدف والمصدر ممثل بشري)، وتسريب الـ prosody والأسلوب الشخصي حتى بعد تغيير الصوت، والحقوق: تحويل صوت لممثل معروف من غير إذن. ولازم أي استخدام داخلي يتوثق ويتحكم فيه بالـ access control ووسم الناتج بـ watermark.

**سؤال متابعة:** لو استخدمت VC لتوليد data للـ STT، ازاي تتأكد إن الموديل ما بيتعلمش يكتشف الـ VC artifacts بدل ما يتعلم اللهجة؟ (ablation: تدريب مع وبدون، واختبار على data حقيقي بس).

### س53. الـ TTS في الـ production بيغلط بطرق مختلفة عن الـ demo: كلمات مهلوسة، كلمات متخطاة، أرقام غلط، أصوات غريبة. ازاي تبني guardrails على الـ output؟

**الإجابة النموذجية:**
الـ failure modes الشائعة في الموديلات الحديثة: (1) hallucination: كلمات أو مقاطع مش في النص. (2) تخطي كلمات أو توقف مبكر. (3) قراءة أرقام وتواريخ ومبالغ غلط (أخطر حاجة في الـ banking). (4) تكرار أو run-on. (5) تغير مفاجئ في الصوت أو الـ pitch أو صوت "شيطاني" عند الـ chunk boundaries. (6) ضوضاء أو clicks أو صمت طويل. (7) نطق غلط لأسماء. (8) تشكيل غلط بيغيّر المعنى.

الـ guardrails: (1) ASR verification loop: تعمل STT على الناتج وتقارن بالنص المطبّع (بعد normalization متطابق) وتحسب WER أو character error؛ لو فوق threshold تعيد التوليد بـ seed مختلف أو بـ CFG مختلف، ولو فشل مرتين ترجع لموديل fallback مستقر (أقل جودة وأكتر موثوقية). التكلفة إضافة STT سريع (Parakeet أو faster-whisper) في المسار، وده مقبول في الـ offline generation وصعب في الـ streaming، فهناك بتعمله على الـ chunk الأول فقط أو تعتمد على الـ heuristics. (2) للأرقام والمبالغ: تحويلها لكلمات في الـ text normalization قبل الموديل (الموديل ما يشوفش أرقام أبدًا) وتتحقق بالـ ASR إن الكلمات دي موجودة. (3) الـ duration sanity: الطول الناتج مقارنة بالمتوقع من عدد الحروف (حوالي 12-16 حرف عربي في الثانية)، لو خرج عن المدى بمقدار كبير أعد التوليد. (4) audio checks: clipping، صمت أطول من ثانيتين، مستوى loudness. (5) speaker similarity مع الـ reference للتأكد إن الصوت ما اتغيرش. (6) قاموس نطق ومسموحات مسبقة لأسماء المنتجات.

والأهم القياس المستمر: نسبة إعادة التوليد، نسبة الـ fallback، ومراجعة بشرية لعينة من الناتج كل يوم، لأن التدهور بيكون تدريجي بعد تحديثات النص أو الموديل.

**سؤال متابعة:** الـ ASR verification بيرفض جمل صح لأن الـ STT بيغلط في اللهجة. ازاي تفصل بين غلط الـ TTS وغلط الـ STT؟ (استخدام موديلين STT، وthreshold على مستوى الـ phonetic similarity مش الـ exact match، وقياس الـ false rejection rate).

### س54. التحكم في النطق: الـ lexicon والـ phoneme override والـ SSML. ازاي تتعامل مع أسماء المنتجات والأماكن السعودية والأسماء الأجنبية والاختصارات؟

**الإجابة النموذجية:**
مفيش موديل هينطق "العُلا" و"أبها" و"iPhone 15 Pro Max" و"STC" صح من نفسه باتساق. الحل طبقة lexicon قبل الموديل: (1) قاموس نطق للكلمات الخاصة، لكل مدخل: النص كما بيظهر، والصورة المنطوقة (إما بالتشكيل الكامل، أو بحروف عربية صوتية للكلمات الأجنبية، أو phonemes لو الموديل بياخد phonemes). (2) قواعد للاختصارات: "STC" تتقرا حروف "إس تي سي"، "SAR" تتقرا "ريال"، "KSA" حسب السياق. (3) الأسماء الأجنبية بتتحول لكتابة عربية منطوقة. (4) أدوات الـ homographs: كلمة زي "علم" لها أكتر من نطق حسب المعنى، والـ diacritizer المبني على context بيحل أغلبها، والباقي بالـ override اليدوي.

الـ SSML (أو تنسيق مكافئ) بيدي تحكم على مستوى الجملة: phoneme tag لنطق محدد، say-as للأرقام (digits مقابل cardinal مقابل date)، break للوقفات، prosody للسرعة والـ pitch. الموديلات الـ LLM-style الحديثة غالبًا ما بتدعمش SSML كامل، فبتطبّق معظمه في الـ frontend (تحويل say-as لكلمات، وbreak لصمت مُدرج بعد التوليد) وبتسيب للموديل الـ prosody.

الجزء التنظيمي أهم من التقني: الـ lexicon لازم يكون قابل للتعديل من غير deployment (ملف أو جدول محمّل ديناميكيًا)، وله owner من فريق الـ content مش الـ ML، وله test suite: لكل مدخل جملة اختبار بتتولّد وبتتراجع بالودن أو بالـ ASR، وأي تغيير في الموديل بيتعاد عليه الاختبار. والتشكيل اليدوي للأسماء في الـ lexicon هو أكتر حاجة بتحسّن الانطباع عن الجودة عند العميل السعودي بتكلفة قليلة جدًا.

**سؤال متابعة:** الموديل بياخد نص خام (مش phonemes). ازاي تعمل phoneme override في حالة زي دي؟ (التشكيل الكامل زائد كتابة صوتية، وتدريب الموديل على نسبة من الـ data المكتوبة بالصورة دي عشان يفهمها).

### س55. الـ expressive TTS العربي: ازاي تجمع data وlabels للانفعالات والأساليب، وازاي تتعامل مع الضحك والتنهد والتردد وأصوات غير كلامية؟

**الإجابة النموذجية:**
التعبير بيتعلم من الـ data قبل أي معمارية. في التسجيل: بدل ما تطلب من الممثل "اقرا بغضب" (بيطلع مبالغ فيه)، بتكتب سياقات (جمل من مواقف حقيقية: اعتذار لعميل، ترحيب، تنبيه، خبر سيئ) وبتخليه يمثّل الموقف، وبتسجّل نسخ متعددة بدرجات مختلفة. الـ labels: على مستوى الجملة (style/emotion label وشدة) ويفضّل تكون من أكتر من annotator لأن الاتفاق بيكون واطي؛ والـ labels التلقائية من موديلات emotion recognition ضعيفة للعربي وممكن تعلّم الموديل غلط.

المعمارية: style tokens (GST) أو reference encoder بتتعلم الأسلوب من غير labels صريحة، والـ instruction-based (CosyVoice-Instruct وParler وأمثالها) بتاخد وصف نصي للأسلوب، وده بيحتاج captions للـ data (مولّدة بـ LLM من الـ labels والخصائص الصوتية المقاسة: سرعة، pitch range، energy). في الموديلات الـ LLM-style الـ prompt الصوتي نفسه بينقل الأسلوب، فلو عايز الصوت مبتهج تدي prompt مبتهج من نفس المتكلم.

الأصوات غير الكلامية (ضحك، تنهد، "امم"، "آه"، شهيق): الموديلات الحديثة بتولّدها لو شافتها في التدريب بـ tags صريحة في النص ([ضحك] [تنهد])، وده بيحتاج transcripts فيها الـ tags دي بالموضع الصحيح؛ Orpheus وDia وElevenLabs v3 بيدعموا tags زي دي. في الـ Arabic data المتاح ده نادر جدًا فلازم تسجّله بنفسك، وتعرّف الـ LLM اللي بيكتب الرد إنه يقدر يستخدم الـ tags باعتدال.

الخطر: التعبير الزائد في voice agent بنكي مش مناسب؛ المطلوب غالبًا دفء واتزان و"micro-expressions" (تعاطف خفيف عند الاعتذار)، والقياس بيكون CMOS على الأسلوب المطلوب مقابل الحيادي، واختبار إن الوضوح ما اتأثرش.

**سؤال متابعة:** الـ LLM بيحط [ضحك] في أماكن مش مناسبة. ايه الضوابط؟

### س56. الـ TTS للـ telephony والـ IVR: 8 kHz، الـ loudness، الـ pre-rendered prompts، والـ dynamic content (اسم العميل والمبلغ) جوه جملة ثابتة. ايه اللي بيختلف عن الـ web؟

**الإجابة النموذجية:**
في التليفون الناتج هيتحول لـ 8 kHz G.711 في الآخر، فالجودة العالية بتضيع، لكن الوضوح بيبقى أهم من أي وقت: الأصوات الرفيعة (ف، س، ث) بتتشابه، فبتختار صوت بـ pitch متوسط ونطق واضح، وتتحكم في السرعة (أبطأ شوية من الـ web)، وتعمل downsampling بشكل صحيح (low-pass قبل الـ decimation) وتظبط الـ loudness عند مستوى مناسب للشبكة (الـ carriers عندها حدود، والصوت العالي بيتقص). وتختبر على تليفون حقيقي مش على السماعة، لأن السماعة بتخدع.

الـ pre-rendering: كل الجمل الثابتة (ترحيب، قوائم، رسائل انتظار) بتتولّد مرة وتتراجع بالودن وتتخزن، وده بيلغي الـ latency والمخاطر. الـ dynamic content هو التحدي: "رصيدك الحالي {مبلغ} ريال". فيه ثلاث طرق: (1) توليد الجملة كاملة كل مرة بالـ TTS (أطبع، بس latency ومخاطر أرقام). (2) concatenation: جزء ثابت pre-rendered زائد جزء ديناميكي مولّد وقتها، والمشكلة إن النغمة بتتغير عند الوصلة (الجزء الثابت كان له prosody جملة كاملة). (3) توليد الجملة كاملة مع cache على مستوى الـ template + القيمة الشائعة. الحل العملي: التوليد الكامل مع streaming لما الموديل سريع، والـ concatenation للأنظمة القديمة أو الـ CPU-only، مع تسجيل الجزء الثابت بنبرة "مفتوحة" (مش نهاية جملة) عشان الوصلة تبقى مقبولة.

التفاصيل اللي بتفرق: قراءة الأرقام (رقم الحساب أرقام أرقام، المبلغ ككلمات مع "ريال" و"هللة")، التواريخ الهجرية والميلادية، الوقفات قبل القيم المهمة، وتكرار القيمة مرة تانية لو طويلة. وفي الـ IVR القديم (VoiceXML) الـ TTS بيتنده عبر MRCP، فالـ integration نفسه بيحتاج دعم البروتوكول ده أو gateway.

**سؤال متابعة:** الـ TTS ممتاز على 24 kHz بس على التليفون بيطلع "مكتوم" أكتر من الأصوات المنافسة. ايه الأسباب المحتملة؟ (الـ spectral balance بتاع الصوت نفسه، الـ downsampling، الـ loudness، أو pre-emphasis ناقصة).

### س57. الـ TTS الخفيف على CPU والـ edge: Piper وVITS الصغير وKokoro وMatcha-TTS وStyleTTS 2. ايه المتاح للعربي، وازاي تبني صوت عربي خفيف؟

**الإجابة النموذجية:**
لما الـ GPU مش متاح (أجهزة داخل المنشآت، تطبيقات موبايل، أنظمة embedded، أو لتقليل التكلفة)، الموديلات الـ VITS-style بتكسب: Piper (VITS مبسّط بـ ONNX، ملايين قليلة من الـ parameters، بيطلّع جملة في أجزاء من الثانية على CPU واحد، وعنده أصوات عربية محدودة الجودة بس المعمارية مفتوحة MIT)، وKokoro (حوالي 82M parameter مبني على StyleTTS 2، جودة عالية على CPU، لكن ما بيدعمش العربي)، وMatcha-TTS (flow matching صغير مع vocoder خفيف، بيتدرب بسهولة)، وStyleTTS 2 (جودة ممتازة بس التدريب أعقد). الموديلات الـ LLM-style بالمقارنة بتحتاج GPU لأن الـ decoder بيولّد عشرات الـ tokens في الثانية من موديل بمئات الملايين أو مليارات الـ parameters.

بناء صوت عربي خفيف: (1) data: 3-10 ساعات نظيفة لمتكلم واحد بنص مشكّل بالكامل (لأن الموديلات دي بتاخد phonemes أو حروف بتشكيل، مش نص خام). (2) الـ frontend: diacritizer ثم G2P قواعدي (زي phonetiser بتاع Arabic Speech Corpus أو قواعد مبنية على Buckwalter)، ولازم يكون deterministic وسريع لأنه هيشتغل على CPU كمان. (3) التدريب: VITS أو Matcha مع vocoder خفيف (Vocos أو HiFi-GAN صغير)، ساعات على GPU واحد. (4) الـ export لـ ONNX وقياس RTF على الـ CPU المستهدف مع int8 quantization لو محتاج. (5) الـ streaming: الموديلات دي مش streaming بطبيعتها، فبتقطّع بالجمل.

التنازلات: الـ prosody أقل طبيعية من الـ LLM-style، والـ zero-shot cloning غير متاح، والتحكم بالأسلوب محدود. بس للـ IVR وقراءة الإشعارات وأنظمة كتير، ده كافي وبتكلفة صفر GPU.

**سؤال متابعة:** الـ diacritizer نفسه ممكن يكون موديل ثقيل. ايه الحل لو محتاج كل الـ pipeline على CPU في 200 ms؟

### س58. ازاي تصمم evaluation set وبروتوكول تقييم لـ Arabic TTS بالتحديد؟ ايه الحالات اللي لازم تكون فيه، ومين يقيّم، وازاي تفصل أخطاء التشكيل عن أخطاء الصوت؟

**الإجابة النموذجية:**
الـ MOS العام على جمل عادية بيخبي أغلب مشاكل العربي. الـ evaluation set لازم يكون فيه فئات صريحة: (1) جمل عادية MSA ولهجة. (2) جمل مليانة أرقام ومبالغ وتواريخ (هجري وميلادي) وأوقات ونسب. (3) أسماء أشخاص وأماكن ومنتجات. (4) code-switching. (5) كلمات متشابهة الرسم مختلفة التشكيل في سياق يحدد المعنى (homographs). (6) جمل طويلة ومعقدة نحويًا. (7) أسئلة وتعجب وأوامر (الـ intonation). (8) همزات وتاء مربوطة ووقف على نهاية الكلمة (الـ pausal forms). (9) جمل قصيرة جدًا (كلمة واحدة) لأن الموديلات بتتعب فيها. لكل فئة 20-50 جملة، ونفس الـ set بيتقيّم عليه كل موديل أو نسخة.

فصل الأخطاء: بتقيّم على مرحلتين. الأولى الـ frontend لوحده: تطلّع النص المشكّل والمطبّع، وتقيس دقة التشكيل (DER على مستوى الحرف وWER على مستوى الكلمة) مقابل مرجع بشري، ودقة الـ normalization للأرقام؛ أي خطأ هنا مش ذنب الموديل الصوتي. التانية الصوت من نص مشكّل صح: intelligibility بـ dictation test (مقيّمين بيكتبوا اللي سمعوه وتحسب WER) أو ASR كـ proxy، والـ naturalness بـ MOS أو الأفضل CMOS مقابل baseline، والـ speaker similarity، وتقييم النطق للأسماء بـ checklist ثنائي (صح/غلط).

المقيّمين: لازم يكونوا native في اللهجة المستهدفة (مقيّم مصري هيدي MOS عالي لصوت نجدي فيه أخطاء لهجية ما يسمعهاش)، وعددهم كافي (8-15 لكل جملة على الأقل للـ MOS)، مع attention checks وجمل مرجعية مخفية (تسجيل بشري حقيقي وتسجيل مكسور عمدًا) لمعايرة المقيّمين. والـ CMOS بين نسختين متتاليتين أكتر حساسية من MOS مطلق لتتبع التحسن.

**سؤال متابعة:** الـ MOS الإجمالي 4.3 بس العملاء بيشتكوا. ايه اللي ممكن يكون ناقص في البروتوكول؟

---

## 4. الـ real-time voice agents

### س59. قارن بين الـ cascaded pipeline (VAD ثم STT ثم LLM ثم TTS) والـ end-to-end speech-to-speech models. امتى تختار أنهي؟

**الإجابة النموذجية:**
**الـ cascaded:** كل مرحلة موديل منفصل ومتخصص. المميزات: تقدر تختار best-of-breed لكل مرحلة، والنص في النص بيديك كل حاجة: logging، guardrails، RAG، tool calling، تقييم واضح، وتبديل أي component من غير ما تعيد تدريب حاجة. العيوب: الـ latency بتتجمع من كل مرحلة، الأخطاء بتتراكم (غلطة في الـ STT بتوصل للـ LLM كحقيقة)، وبتضيّع الـ paralinguistics (نبرة، تردد، ضحك، انفعال) لأن الـ LLM بيشوف نص بس.

**الـ speech-to-speech (GPT-4o Realtime، Moshi، Qwen-Omni وأشباهها):** الموديل بياخد audio tokens وبيطلع audio tokens (غالبًا مع text كـ inner monologue). المميزات: latency أقل، بيحافظ على النبرة ويرد بنبرة مناسبة، والـ turn-taking والـ interruptions طبيعية أكتر (Moshi full-duplex). العيوب: أقل قابلية للتحكم، أصعب في الـ guardrails والتقييم، الـ reasoning أضعف من الـ text LLMs من نفس الحجم، الـ tool calling أقل نضجًا، وفي العربي واللهجات الدعم ضعيف أو غير موجود في الـ open-source، ومش هتقدر تعمل fine-tuning على domain بسهولة.

**الاختيار:** لـ enterprise وregulated domains (بنوك، حكومة) اللي محتاجة auditability وon-prem ودعم عربي مضبوط، الـ cascaded هو الخيار العملي حاليًا، مع تحسين الـ latency بالـ streaming في كل مرحلة. الـ S2S مناسب للـ consumer experiences بالإنجليزي أو لما الـ paralinguistics جزء من القيمة. الاتجاه المستقبلي hybrid: موديل سمع ونطق موحد مع text LLM قوي في المنتصف (thinker-talker)، والـ open-source العربي محتاج data dialogue ضخمة عشان يوصل.

**سؤال متابعة:** ازاي تدخّل معلومة زي "المستخدم صوته متضايق" للـ LLM في cascaded pipeline؟ (audio event/emotion classifier بيضيف tags للـ prompt).

### س60. حلّل الـ latency budget لـ voice agent من نهاية كلام المستخدم لبداية صوت الرد. ايه الأرقام المستهدفة وفين الوقت بيروح؟

**الإجابة النموذجية:**
البشر في المحادثة بيردوا في حدود 200 إلى 500 ms، وأي حاجة فوق ثانية بتحس إنها بطيئة. الهدف العملي: voice-to-voice p50 أقل من 800 ms وp95 أقل من 1.5 ثانية.

**تفصيل الـ budget (تقريبي):**
1. **الـ endpointing:** أكبر مكوّن. لو الـ silence threshold 700 ms فانت دفعت 700 ms قبل ما تبدأ أي حاجة. الحل: semantic endpointing يقلل الـ threshold لـ 200 إلى 400 ms لما الجملة تبان مكتملة.
2. **الـ STT final:** مع streaming STT الـ partials موجودة، والـ final بعد الـ endpoint 100 إلى 300 ms. مع Whisper غير الـ streaming ممكن 500 ms إلى ثانية.
3. **الـ LLM TTFT (time to first token):** 200 إلى 600 ms حسب الموديل وطول الـ prompt. الـ prompt بيطول مع المكالمة فالـ TTFT بيزيد، والحل: prefix caching للـ system prompt وRAG context، summarization للـ history، موديل أصغر للمحادثة وأكبر للقرارات الصعبة.
4. **الـ TTS TTFB:** 100 إلى 300 ms لأول chunk، بشرط sentence-level streaming.
5. **الشبكة والـ playback buffer:** 50 إلى 200 ms (WebRTC أقل من WebSocket، والـ jitter buffer بيضيف).

**التحسينات المهمة:** streaming في كل حلقة (متستناش نهاية أي مرحلة)، speculative LLM call على الـ partial transcript وإلغاؤه لو اتغير، colocation لكل الـ services في نفس الـ region/VPC (كل hop شبكة بيضيف 20 إلى 50 ms)، GPU warm وbatching معقول، وقياس كل مرحلة بـ timestamps موحدة. ولازم يقول إن الـ average مضلل، الـ p95 هو اللي بيحدد التجربة.

**سؤال متابعة:** لو الـ p95 عالي والـ p50 كويس، فين تدور الأول؟ (GPU queueing تحت الضغط، طول الـ prompt في المكالمات الطويلة، وtool latency).

### س61. الـ barge-in (المستخدم بيقاطع الـ agent): ازاي تنفذه صح؟

**الإجابة النموذجية:**
الخطوات:
1. **الـ AEC شغال** وإلا هتقاطع نفسك بصوتك. في الـ browser بيجي مع WebRTC، وفي telephony لازم تعمله server-side أو تعتمد على الـ echo cancellation في الـ gateway.
2. **اكتشاف الكلام أثناء الـ playback** بالـ VAD، لكن مش أي صوت يقاطع: اطلب min speech duration (200 إلى 500 ms) أو انتظر الـ STT partial يطلع كلمة فعلية، عشان الكحة والضوضاء ميقطعوش الـ agent.
3. **تمييز الـ backchannels:** "اه"، "تمام"، "ماشي" مش مقاطعة، دي إشارات متابعة. موديل صغير أو قائمة كلمات مع طول قصير بتفلترها.
4. **الإلغاء الفوري:** وقّف الـ TTS generation، امسح الـ audio buffer في الـ client (مش بس في الـ server، لأن ممكن يكون فيه ثانية أو اتنين audio محملة)، ووقّف الـ LLM stream.
5. **تحديث الـ conversation history:** أهم خطوة بيتنساها كتير: الـ LLM لازم يعرف إن الرد اتقطع وفين بالظبط. باستخدام الـ playback position والـ word timestamps من الـ TTS تحدد الجزء اللي المستخدم سمعه فعلًا، وتحط في الـ history الجزء ده بس مع علامة interrupted، وإلا الـ LLM هيفترض إن المستخدم سمع الرد كله.
6. **القياسات:** false barge-in rate (قطع من غير كلام حقيقي) وmissed barge-in rate وlatency من بداية كلام المستخدم لسكوت الـ agent (الهدف أقل من 300 ms).

**سؤال متابعة:** لو المستخدم بيتكلم والـ agent لسه بيفكر (قبل ما يبدأ ينطق)، بتعمل ايه بالـ LLM request الجارية؟

### س62. الـ turn detection: ليه الـ fixed silence timeout مش كافي، وايه البدائل؟

**الإجابة النموذجية:**
الـ fixed timeout بيحط في مأزق: لو قصير (300 ms) هتقطع المستخدم وهو بيفكر أو بيقرأ رقم بطاقة أو بيقول "أنا عايز... اممم... أحوّل فلوس"، ولو طويل (ثانية) كل رد هيتأخر ثانية والتجربة تبقى بطيئة. الناس بتسكت جوه الجملة أكتر من ما بتسكت بين الجمل أحيانًا.

**البدائل:**
1. **الـ semantic turn detection على النص:** موديل classifier (أو LLM صغير) بياخد الـ partial transcript ويتوقع هل الجملة مكتملة. الـ LiveKit turn detector مثال. لو مكتملة، قلل الـ timeout لـ 200 ms، لو ناقصة (بتنتهي بـ "و"، "عشان"، أو رقم غير مكتمل)، زوّده لـ 1.5 ثانية.
2. **الـ acoustic cues:** نزول الـ pitch في الآخر، إطالة آخر مقطع، الـ filler words ("يعني"، "اممم") بتدل على استمرار. موديلات زي Smart Turn بتشتغل على الـ audio مباشرة.
3. **الدمج:** VAD + semantic + acoustic في decision function، مع adaptive thresholds حسب الـ state في الحوار (لو الـ agent سأل عن رقم، استنى أكتر).
4. **الـ LLM نفسه:** بعض الأنظمة بتخلي الـ LLM يقرر "أرد ولا أستنى" كجزء من الـ output، بس ده أبطأ.

في العربي: الـ fillers مختلفة حسب اللهجة، والموديلات الجاهزة متدربة على الإنجليزي غالبًا، فمحتاج تدرب turn detector على transcripts عربية أو تعمل fine-tuning. والقياس لازم يكون مزدوج: latency ونسبة القطع الخاطئ مع بعض، لأن تحسين واحد بيضر التاني.

**سؤال متابعة:** ازاي تجمع training data لـ turn detector من مكالمات حقيقية؟ (من الـ human-human calls بالـ diarization: نقاط تبادل الأدوار الحقيقية كـ positives، والـ pauses داخل الدور كـ negatives).

### س63. الـ transport: WebRTC مقابل WebSocket مقابل telephony (SIP/RTP). ايه الاختيار وليه؟ واشرح jitter buffer والـ packet loss والـ Opus.

**الإجابة النموذجية:**
**الـ WebRTC:** بيشتغل على UDP (SRTP)، مصمم للـ real-time: jitter buffer adaptive، packet loss concealment، congestion control، NAT traversal بـ ICE/STUN/TURN، وفي المتصفح بيدي AEC وNS وAGC مجانًا، والـ codec الافتراضي Opus. الأفضل للـ browser والموبايل. تعقيده في الـ signaling والـ infra (محتاج SFU/media server زي LiveKit أو mediasoup).

**الـ WebSocket:** TCP، فأي packet ضايع بيوقف كل اللي بعده (head-of-line blocking)، والـ latency بتتضاعف تحت الـ loss، ومفيش audio processing مدمج. بسيط جدًا ومناسب لـ server-to-server (بين الـ orchestrator وخدمة STT مثلًا) أو للـ prototypes، ومناسب لما الشبكة مضمونة.

**الـ telephony:** SIP للـ signaling وRTP للـ media، بـ G.711 (μ-law/A-law عند 8 kHz، 64 kbps) أو G.722 أحيانًا. بتوصله من خلال SIP trunk من شركة الاتصالات لـ media server (FreeSWITCH، Asterisk، أو LiveKit SIP) أو CPaaS (Twilio وأشباهه، وده غير مناسب للـ on-prem). في السعودية غالبًا SIP trunk محلي مع متطلبات data residency.

**الـ Opus:** codec للكلام والموسيقى، من 6 لـ 510 kbps، frames من 2.5 لـ 60 ms (20 ms الشائع)، بيدعم FEC وPLC، وبيشتغل من narrowband لـ fullband. الـ frame size بيحدد جزء من الـ latency.

**الـ jitter buffer:** الـ packets بتوصل بفواصل غير منتظمة، فالـ buffer بيخزن شوية قبل التشغيل عشان الصوت يبقى سلس. أكبر buffer = صوت أنعم لكن latency أعلى. الـ adaptive jitter buffer بيغير حجمه حسب الشبكة. الـ PLC بيملى الـ packets الضايعة بتقدير. لازم يعرف إن الـ audio اللي واصل للـ STT عدى على كل ده، وإن الـ packet loss بيظهر في الـ transcript كحروف ناقصة.

**سؤال متابعة:** ليه ممكن تحتاج TURN server حتى لو الـ deployment داخلي؟ وامتى تستخدم WebSocket بين microservices بدل gRPC streaming؟

### س64. System design: صمم voice agent لمركز اتصال بنك سعودي، 1000 مكالمة متزامنة، on-prem، لهجة نجدية/حجازية مع code-switching إنجليزي.

**الإجابة النموذجية:**
المرشح القوي بيبدأ بالأسئلة: ايه الـ use cases (استعلام رصيد، بطاقة ضايعة، تحويل)؟ ايه الـ containment target؟ الـ SLO للـ latency؟ الـ hardware المتاح؟ بعد كده:

**الـ architecture:**
- **طبقة الـ telephony:** SIP trunks ثم SBC ثم media server (FreeSWITCH أو LiveKit SIP) بيحوّل RTP/G.711 لـ audio streams للـ agent workers. Recording مع consent announcement.
- **الـ agent orchestrator:** لكل مكالمة worker (Pipecat أو LiveKit Agents أو framework داخلي) بيدير VAD وendpointing وbarge-in والـ state machine بتاع الحوار، وبيكلم الخدمات الثلاثة بـ gRPC streaming.
- **خدمة الـ STT:** streaming Conformer RNN-T/CTC (NeMo) على Triton بـ sequence batching، متدرب على telephony 8 kHz وعلى اللهجتين مع code-switching، مع hotword biasing لأسماء المنتجات، وITN للأرقام. الأرقام الحساسة (رقم البطاقة، OTP) تتجمع بـ DTMF مش بالصوت لأنها أدق وأأمن.
- **خدمة الـ LLM:** موديل عربي قوي open-weights بعد fine-tuning على vLLM مع prefix caching، structured outputs للـ tool calls، RAG على سياسات البنك مع citations، وguardrails (topic restriction، PII، prompt injection عبر الصوت).
- **خدمة الـ TTS:** صوت brand ثابت fine-tuned streaming على Triton، مع cache للجمل المتكررة (الترحيب، القوائم) كـ pre-rendered audio.
- **الـ integration:** tool calls لأنظمة الـ core banking عبر API gateway مع authentication (OTP عبر SMS، أو voice biometrics + liveness كعامل إضافي مش وحيد).
- **الـ human handoff:** تحويل warm للموظف مع ملخص المكالمة وتحويل الـ context.

**الـ capacity planning:** قيس عمليًا عدد الـ concurrent streams لكل GPU لكل خدمة عند الـ p95 المستهدف (تقريبي: STT streaming بالـ batching مئات الـ streams على GPU واحد، TTS عشرات لمئة، LLM حسب tokens/sec وطول الـ context)، ثم 1000 مكالمة × نسبة الوقت اللي كل خدمة بتشتغل فيه فعليًا (المستخدم بيتكلم 40%، الـ agent 40%، صمت 20% تقريبًا) + headroom 30% + N+1 للـ failover. VAD على CPU. الـ media server نفسه محتاج CPU كتير للـ transcoding.

**الـ observability:** trace لكل turn بـ timestamps لكل مرحلة، sampling للمكالمات للـ WER review، dashboards للـ latency p50/p95، containment، handoff rate، barge-in metrics، وalerts على GPU queue depth.

**الـ failure modes:** انقطاع أي خدمة يبقى fallback لـ IVR بالـ DTMF أو تحويل مباشر لموظف، STT بيغلط في الأرقام يبقى DTMF وتأكيد بالإعادة، endpointing بيقطع كبار السن يبقى adaptive timeouts، TTS بينطق الأسماء غلط يبقى lexicon overrides، LLM بيهلوس سياسة يبقى RAG-only answers مع refusal، echo على الـ telephony يبقى server-side AEC وhalf-duplex fallback، والمكالمات الطويلة بتكبّر الـ context يبقى summarization.

**الامتثال:** PDPL وتعليمات SAMA، data residency كاملة داخل الـ data center، تشفير التسجيلات، retention policy، وmasking للـ PII في الـ transcripts والـ logs.

**سؤال متابعة:** لو الـ GPUs محدودة (8 GPUs بس)، ايه اللي تضحي بيه الأول؟ وازاي تعمل load test بمكالمات synthetic قبل الإطلاق؟

### س65. ازاي تقيّم voice agent end-to-end، مش بس الـ components؟

**الإجابة النموذجية:**
الـ component metrics (WER، TTS MOS، LLM accuracy) ضرورية لكن مش كافية، لأن agent ممكن يكون كل مكوّن فيه ممتاز والتجربة سيئة بسبب الـ turn-taking.

**مقاييس على مستوى المحادثة:** task completion rate (المستخدم حقق هدفه؟)، containment rate (المكالمة اتحلت من غير موظف)، average handling time، latency per turn (p50/p95)، interruption metrics (false barge-ins، والمستخدم بيقاطع كتير = علامة إن الـ agent بيطوّل أو بيغلط)، repetition rate (المستخدم بيعيد كلامه = STT أو فهم ضعيف)، hangup rate في أول 30 ثانية، وCSAT في الآخر.

**التقييم الـ offline قبل الإطلاق:** (1) regression suite من مكالمات حقيقية مسجلة بتتعاد على الـ pipeline كلها مع expected outcomes. (2) simulated users: LLM بيلعب دور العميل بـ personas وأهداف، وصوته بيتولد بـ TTS بلهجات مختلفة وضوضاء وتليفون، وبيتكلم مع الـ agent فعلًا بالصوت، وبعدين LLM-as-judge بيقيّم الـ transcript على rubric (صحة المعلومة، الالتزام بالسياسة، اللباقة). ده بيسمح بآلاف المحادثات ليلًا. (3) adversarial tests: prompt injection صوتي، لهجات غير متوقعة، صمت طويل، كلام متداخل.

**بعد الإطلاق (online):** A/B testing بين نسخ الـ agent مع guardrails، وreview بشري لعينة يومية من المكالمات مع تصنيف الأخطاء (STT، endpointing، LLM، TTS، integration) عشان تعرف فين تستثمر. الأخطاء الخاصة بالصوت (أرقام، أسماء، تواريخ) تبقى categories منفصلة.

**سؤال متابعة:** ازاي تفرّق بين خطأ سببه الـ STT وخطأ سببه الـ LLM لما المستخدم يشتكي إن الـ agent "مش فاهم"؟ (تسمع الـ audio مع الـ transcript مع الـ LLM input في نفس الأداة).

### س66. الـ tool calling في الـ voice agents: ايه اللي بيختلف عن النص، وازاي تتعامل مع الـ tools البطيئة؟

**الإجابة النموذجية:**
الفرق الأساسي إن الصمت في الصوت مؤلم: ثانيتين صمت في chat عادي، في الصوت المستخدم بيقول "ألو؟". فالمبادئ:

1. **الـ filler speech:** لو الـ tool هياخد أكتر من 500 ms، الـ agent يقول جملة طبيعية ("ثانية واحدة أشوف الحساب") بتتولد فورًا (pre-rendered أو من الـ LLM في نفس الوقت مع الـ tool call)، ومش بتتكرر بنفس الصيغة كل مرة.
2. **التنفيذ الـ speculative والمتوازي:** ابدأ الـ tool call من الـ partial transcript لو الـ intent واضح، وشغّل الـ tools المستقلة بالتوازي.
3. **الـ voice UX للـ outputs:** ما تقراش جدول من 10 صفوف. الـ LLM لازم يلخص ويقدم 2 إلى 3 خيارات، ويقرأ الأرقام بشكل مناسب للـ TTS (تجميع أرقام الحساب في مجموعات)، والـ TN لازم يبقى متسق.
4. **التأكيد الصوتي للأفعال الحساسة:** قبل التحويل يعيد المبلغ والمستفيد ويستنى تأكيد صريح، مع التعامل مع عدم اليقين في الـ STT للـ slot values (confidence واطي على المبلغ يبقى أعد السؤال، أو DTMF).
5. **الـ async tools:** لو العملية بتاخد وقت طويل (فتح تذكرة)، الـ agent يكمل الحوار ويرجع للنتيجة لما توصل، أو يوعد بتواصل لاحق.
6. **الـ structured outputs والـ schema validation** عشان الـ LLM ميبعتش arguments غلط، وretries مع رسائل واضحة للمستخدم.

**سؤال متابعة:** ازاي تتعامل مع المستخدم اللي بيتكلم أثناء الـ filler speech؟ (نفس الـ barge-in logic مع إلغاء الـ filler بس مش الـ tool call).

### س67. الـ full-duplex models (زي Moshi): ازاي بتشتغل وايه المشكلة اللي بتحلها؟

**الإجابة النموذجية:**
الأنظمة العادية half-duplex: إما المستخدم بيتكلم أو الـ agent، والتبديل بيحصل بالـ endpointing. Moshi (Kyutai) بيمثل المحادثة كـ streams متوازية بنفس الـ frame rate (12.5 Hz بـ Mimi codec): stream للمستخدم وstream للنظام، والموديل بيتوقع كل frame للنظام (audio tokens + text tokens كـ inner monologue بتسبق الصوت وبتحسّن جودة اللغة) وهو بيسمع الـ stream بتاع المستخدم في نفس اللحظة. فمفيش مفهوم "دور"، الموديل بيقدر يعمل backchannel ("اها")، يتقاطع، يتوقف لما المستخدم يتكلم، ويتكلم فوق المستخدم لو لازم، بـ latency نظرية حوالي 160 إلى 200 ms.

**اللي بيحله:** الـ turn-taking الطبيعي، وده أكبر مشكلة UX في الـ cascaded systems، وكمان الاستفادة من الـ paralinguistics.

**اللي مش بيحله:** الـ reasoning والدقة (الموديل صغير نسبيًا وبيتدرب على dialogues synthetic كتير)، الـ tool calling، التحكم، والدعم اللغوي (إنجليزي وفرنسي أساسًا). للعربي هتحتاج codec يتدرب أو يتقيّم على العربي، وpretraining على آلاف الساعات من الكلام العربي، وdialogue data full-duplex مش موجودة، فده اتجاه بحثي أكتر منه production حاليًا. Kyutai نفسهم طلعوا Unmute كـ cascaded system بـ streaming STT/TTS مع نفس فلسفة الـ low latency، وده مؤشر إن الـ cascaded لسه عملي.

**سؤال متابعة:** ايه هو الـ inner monologue في Moshi وليه بيحسّن الجودة؟

### س68. الـ frameworks بتاعة الـ voice agents (LiveKit Agents وPipecat وأمثالهم) مقابل بناء الـ orchestration بنفسك: ايه اللي بيدوهولك، وايه اللي هتبنيه في كل الأحوال، وايه معايير الاختيار للـ on-prem؟

**الإجابة النموذجية:**
الـ frameworks دي بتحل المشاكل المملة اللي بتاخد شهور: الـ transport (WebRTC وSIP وWebSocket)، الـ audio plumbing (resampling، frames، jitter)، الـ VAD والـ turn detection والـ interruption handling، الـ pipeline بين STT وLLM وTTS بشكل streaming، وplugins جاهزة لعشرات الـ providers والموديلات المفتوحة. LiveKit Agents مبني على LiveKit server (مفتوح المصدر وينفع self-hosted) مع SIP gateway وturn detector model خاص بيهم؛ Pipecat من Daily مبني كـ pipeline من frame processors مرن جدًا ومعاه Smart Turn model مفتوح؛ وفيه SaaS كاملة (Vapi وRetell وأمثالهم) مش مناسبة للـ on-prem.

اللي هتبنيه في كل الأحوال: منطق الحوار والـ state، الـ tools والـ integrations مع أنظمة الشركة، الـ guardrails، الـ evaluation والـ observability، الـ prompt engineering باللهجة، الـ lexicon، وسياسات الـ fallback والـ handoff. يعني الـ framework بيوفر 30-40% من الشغل، بس الـ 60% الباقية هي اللي بتفرق في الجودة.

معايير الاختيار للـ on-prem: (1) الـ license (Apache 2.0 لـ LiveKit وBSD-2 لـ Pipecat) وإمكانية التشغيل بدون أي خدمة سحابية بما فيها الـ turn detector والـ VAD. (2) دعم SIP الحقيقي مع الـ PBX الموجود. (3) قابلية استبدال أي مكوّن بموديل داخلي (STT وTTS وLLM على Triton أو vLLM) من غير hack. (4) النضج في الـ interruption handling والـ backpressure. (5) الـ observability المدمجة. (6) حجم الـ community وسرعة الإصلاحات. (7) إن الـ framework ما يفرضش شكل الحوار (بعضها بيفترض أن كل turn نص واحد ورد واحد).

المرشح القوي بيقول إنه غالبًا يبدأ بـ framework عشان يوصل لـ pilot بسرعة، وبيحافظ على abstraction رفيعة فوقه عشان يقدر يستبدله أو يعدّل جواه لما يوصل لحدوده، وبيذكر حدود شافها فعلًا (مثلًا التحكم الدقيق في الـ barge-in أو الـ latency في الـ SIP path).

**سؤال متابعة:** ايه الحاجة الأولى اللي بتضطر تعدّلها جوه أي framework بعد أول pilot حقيقي؟

### س69. إدارة الـ state في الـ voice agent: الـ turn-taking state machine، والـ partial transcripts، وذاكرة المكالمة الطويلة، وايه اللي بيتحفظ لو المكالمة اتقطعت.

**الإجابة النموذجية:**
الـ agent محتاج state machine صريح على الأقل بالحالات: listening، user speaking، processing (STT final ثم LLM)، speaking، interrupted، tool waiting، وtransfer. كل انتقال بيتسجل بـ timestamp، وده أساس الـ debugging والـ metrics. الغلط الشائع إن الـ state ضمني موزّع على callbacks، فبتحصل race conditions: الـ TTS بيبدأ يتكلم بعد ما المستخدم بدأ يتكلم تاني، أو رد قديم بيتشغل بعد الـ barge-in.

الـ partial transcripts: بتستخدمها لثلاث حاجات، الـ turn detection (النص اللي لسه ناقص)، الـ speculative LLM calls (تبدأ توليد رد على الـ partial وتلغيه لو الـ final اختلف)، والـ UI. لكن ما تحطهاش في الـ conversation history إلا لما تبقى final، وتتعامل مع الحالة اللي الـ final بييجي فيها مختلف عن الـ partial بشكل جوهري.

ذاكرة المكالمة: الـ context بيكبر مع كل turn، والـ TTFT بيزيد، فمكالمة 20 دقيقة ممكن توصل لآلاف الـ tokens. الحلول: prefix caching في الـ LLM server (عشان الـ history المشترك ما يتحسبش تاني)، وsummarization لما يعدي حد معين مع الاحتفاظ بالـ slots المهمة (رقم العميل، الطلب، القرارات) في structured state منفصل مش في النص فقط، وقطع الـ tool outputs الكبيرة بعد استخدامها.

لو المكالمة اتقطعت: الـ structured state (مين المتصل، فين وصلنا، ايه اللي اتنفذ) بيتحفظ في store خارجي (Redis أو DB) بمفتاح المكالمة، عشان لو اتصل تاني خلال دقايق الـ agent يقدر يكمل ("كنا بنكمل طلب تغيير العنوان، تحب نكمل؟")، وتتأكد إن الـ tools اللي بتغيّر حاجة idempotent عشان ما تتنفذش مرتين.

**سؤال متابعة:** المستخدم قاطع الـ agent في نص جملة. ايه اللي المفروض يتسجل في الـ history: الجملة كاملة اللي الـ LLM ولّدها، ولا الجزء اللي اتسمع فعلًا؟ وليه ده بيفرق في الرد التالي؟

### س70. الـ LLM بيستقبل transcript فيه أخطاء STT. ازاي تخلي المرحلة دي متسامحة مع الأخطاء؟ N-best، الـ phonetic matching للأسماء، واستراتيجيات التأكيد.

**الإجابة النموذجية:**
الخطأ الشائع إنك تعامل الـ transcript كنص مضمون. بدل ده: (1) في الـ prompt، تقول للـ LLM صراحة إن النص جاي من STT وممكن يكون فيه أخطاء صوتية وأسماء مكتوبة غلط، وإنه يفسّر بالمعنى الأقرب ويطلب توضيح لو فيه لبس مهم. (2) تبعت N-best (أفضل 3-5 hypotheses) بدل hypothesis واحدة لما الـ confidence واطي، والـ LLM بيقدر يختار المعنى المتسق مع السياق. (3) الأسماء والـ IDs: بدل ما تعتمد على الـ LLM في مطابقة "محمد العتيبي" مع قاعدة العملاء، تعمل phonetic/fuzzy matching (بالـ edit distance على الحروف بعد normalization، أو على تمثيل صوتي للعربي) مع قائمة المرشحين من الـ CRM، وتخلي الـ tool ترجّع المرشحين والـ LLM يأكد. (4) الـ hotword biasing في الـ STT بأسماء المنتجات والفروع وقوائم الـ menu اللي متوقعة في الـ state الحالية (contextual biasing ديناميكي حسب مرحلة الحوار). (5) استراتيجية التأكيد المتدرجة: تأكيد صريح للحاجات الحرجة (المبالغ، التحويلات، الإلغاء)، وتأكيد ضمني للعادي ("تمام، هحدّث العنوان لحي النرجس")، بحيث المستخدم يصحح لو غلط من غير ما كل turn يبقى سؤال.

كمان: قياس الـ intent accuracy end-to-end (من الصوت) مش على النص المكتوب، لأن الـ LLM ممكن يكون ممتاز على النص الصح وفاشل على النص المغلوط، وتبني test set من transcripts حقيقية بأخطائها.

**سؤال متابعة:** الـ STT كتب "ألغي" بدل "أبغى" (وده فرق كبير في المعنى). ايه الطبقات اللي المفروض تمسك الغلط ده قبل ما يتنفذ إلغاء؟

### س71. جمع أرقام وIDs وتواريخ عبر الصوت (رقم هوية، IBAN، رقم جوال، تاريخ ميلاد): ايه الصعوبات في العربي وازاي تصممها؟

**الإجابة النموذجية:**
الأرقام أسوأ حاجة في الـ voice agents: (1) القراءة بتختلف: "خمسة صفر خمسة" مقابل "خمسمية وخمسة" مقابل "خمسة وخمسين"، والناس بتجمّع الأرقام بطرق مختلفة. (2) العربي فيه تذكير وتأنيث وحالات إعرابية في الأعداد، واللهجات بتنطقها مختلف ("ثنين" و"اثنين" و"تنين"، "تسعة" و"تسعه"). (3) الـ STT بيخلط بين أرقام متشابهة صوتيًا (سبعة وتسعة في الضوضاء). (4) الـ ITN بيقرر يجمّع الأرقام ولا يفصلها. (5) الأرقام الطويلة الناس بتقولها بسرعة وبتغلط هي نفسها.

التصميم: (1) لو متاح، DTMF أوّلًا للأرقام الطويلة والحساسة (رقم الهوية والـ IBAN): أدق وأأمن وما بيتسجلش في الـ transcript. (2) لو صوت: تطلب الرقم على أجزاء (أول 4 أرقام ثم الباقي)، وتطبّق validation فورية (رقم الهوية السعودية 10 أرقام بيبدأ بـ 1 أو 2، الجوال 05 وبعده 8 أرقام، الـ IBAN فيه checksum بيكشف أغلب الأخطاء)، وتعيد الرقم على المستخدم بقراءة مجمّعة واضحة (كل رقم منفصل مع وقفات بين المجموعات) وتستنى تأكيد. (3) في الـ STT: مسار خاص للأرقام، يعني لما الـ state بيقول "مستني رقم"، تشغّل biasing على كلمات الأرقام أو موديل/grammar مقيّد، وتطلّع الأرقام بصيغة digits من الـ ITN بدون تجميع. (4) التواريخ: تسأل عن الشهر والسنة منفصلين لو محتاج، وتتأكد من الهجري/الميلادي. (5) الـ TTS: قراءة الأرقام رقم رقم بسرعة معتدلة، ولو الرقم بيتكرر تسمح بـ "كرر".

القياس: دقة الرقم الكامل (كل الأرقام صح) بعد التأكيد، وعدد المحاولات لكل رقم، ونسبة التحويل لـ DTMF أو لموظف. الـ target الواقعي للصوت فقط غالبًا أقل من 95% من أول محاولة، وده اللي بيبرر الـ DTMF.

**سؤال متابعة:** المستخدم قال "صفر خمسة خمسة ثم واحد اثنين ثلاثة أربعة خمسة ستة سبعة" والـ STT طلّع 0551234567. ازاي تتحقق إن الرقم "صح" مش بس "صالح"؟ (الـ validation بتضمن الصلاحية بس، فلازم تأكيد المستخدم أو مطابقة مع الـ CRM).

### س72. الـ prompt والـ output design لما الـ LLM بيكتب كلام هيتنطق: ايه اللي بيختلف عن الـ chatbot النصي، وازاي تتحكم في اللهجة والطول والتنسيق؟

**الإجابة النموذجية:**
الرد اللي هيتسمع مختلف عن اللي هيتقرا: (1) الطول: جملة أو اتنين لكل turn، لأن المستخدم ما يقدرش "يعمل scroll" في الصوت، والرد الطويل بيدفع للمقاطعة. (2) مفيش markdown ولا قوائم ولا جداول ولا emojis؛ لازم تمنعها في الـ prompt وتنضّفها بعد التوليد كمان (post-processing يشيل الرموز والعناوين). (3) الأرقام والاختصارات بصيغة منطوقة أو بصيغة الـ frontend بيفهمها. (4) الترقيم بيتحول لـ prosody: الفاصلة والنقطة والاستفهام بيغيروا النغمة، فالـ LLM لازم يكتب ترقيم سليم وجمل قصيرة قابلة للـ chunking. (5) اللهجة والسجل: تحدد اللهجة (سعودي بيض أو حسب المستخدم) ومستوى الرسمية، وتمنع الـ LLM إنه يبدّل لـ MSA فجأة أو للإنجليزي؛ الموديلات بتنزلق للـ MSA لأنه أغلب data التدريب، فبتحط أمثلة كتير من الردود باللهجة المطلوبة في الـ prompt (few-shot) وتقيس نسبة الردود اللي خرجت عن اللهجة. (6) التعامل مع الصمت والمقاطعة: تعليمات للـ LLM لما الـ transcript فاضي أو مقطوع. (7) "لا تكرر ما قاله المستخدم بالكامل" لأن التكرار بيطوّل الـ turns.

الـ structured output: الرد بيتقسم لجزء منطوق وجزء ديناميكي (tool calls، state updates، إشارة handoff)، بـ format ثابت (JSON للجزء غير المنطوق) مع streaming للنص المنطوق أولًا عشان الـ TTS يبدأ بدري. والـ LLM لازم يتدرب أو يتوجه إنه يبدأ بالجملة المنطوقة قبل أي حاجة تانية.

المرشح القوي بيذكر إن الـ prompt ده بيتقيّم بـ test suite من الحوارات، والـ metrics: متوسط طول الرد بالثواني بعد الـ TTS، نسبة الالتزام باللهجة، ونسبة الردود اللي فيها رموز اتشالت.

**سؤال متابعة:** ازاي تعمل chunking للنص عشان الـ TTS يبدأ بدري من غير ما تقطع في نص وحدة معنى؟ (على علامات الترقيم مع حد أدنى للطول، وتأجيل الأرقام والأسماء لحد ما تكتمل).

### س73. الـ agent لازم يخدم عربي بلهجات مختلفة وإنجليزي وأردو (عمالة). ازاي تصمم الـ language handling: اكتشاف اللغة، اختيار الموديلات، والـ switching في نص المكالمة؟

**الإجابة النموذجية:**
الخيارات: (1) موديل STT multilingual واحد بيتعامل مع كل اللغات ديناميكيًا (أسهل، وأفضل للـ code-switching، بس أضعف في كل لغة من موديل مخصص). (2) language ID في أول ثواني ثم توجيه لموديل STT وTTS ووصف prompt مخصص لكل لغة (أدق لكل لغة، بس الـ switching في نص المكالمة صعب والـ LID ممكن يغلط). العملي: التحية أول تسأل بلغتين، وتاخد قرار من رد المستخدم (LID على الصوت زائد النص)، وتثبّت اللغة الأساسية للمكالمة، مع إمكانية إعادة الاكتشاف لو الـ STT confidence نزل فجأة أو المستخدم طلب لغة تانية.

للعربي: موديل STT واحد قوي على كل اللهجات أحسن من موديل لكل لهجة، والـ dialect handling بيكون في الـ TTS (صوت ولهجة الرد) والـ LLM (prompt باللهجة). للأردو/الهندية والإنجليزي بلكنات جنوب آسيوية: لازم test sets خاصة، لأن الموديلات المتدربة على إنجليزي أمريكي بتتعب مع اللكنات دي، وده جمهور كبير في السوق.

التفاصيل: الـ TTS لكل لغة بصوت متسق (نفس "الشخصية" بأصوات multilingual لو أمكن)، الـ prompt للـ LLM بيتبدل مع اللغة، والـ tools وقواعد الـ business ثابتة. والـ turn detector والـ VAD لازم يتاختبروا على كل لغة (طول الوقفات ونمط الكلام بيختلف). والقياس بالـ slices: containment وWER وقياس رضا لكل لغة، لأن المتوسط هيخبي إن الأردو أداؤه ضعيف.

**سؤال متابعة:** مستخدم بيتكلم عربي مكسر بلكنة أجنبية. الـ LID بيقول عربي بثقة، بس الـ STT بيغلط كتير. ايه الحل؟ (توسيع data التدريب بعربي بلكنات غير native، والسماح بتبديل للغة المستخدم الأم لو اتكشفت).

### س74. الـ fillers والـ backchannels و"أصوات التفكير": امتى الـ agent يقول "تمام" أو "لحظة" أو "امم"، وازاي تنفذها من غير ما تتكلم فوق المستخدم أو تبان مزيفة؟

**الإجابة النموذجية:**
الإنسان بيملأ الفجوات: "أيوه"، "تمام"، "لحظة أشوف"، وبيعمل backchannels وهو بيسمع ("امم"، "أها"). في الـ agent، الـ fillers بتخفي الـ latency (لما الـ tool هتاخد 3 ثواني) وبتأكد إن الـ agent سمع. لكن لو اتعملت غلط بتبقى مزعجة: filler ثابت كل turn، أو backchannel في نص كلمة المستخدم، أو "لحظة" ثم رد فوري (فبتبان مزيفة).

التنفيذ: (1) الـ acknowledgement fillers بعد نهاية turn المستخدم مباشرة، pre-rendered بصوت الـ agent نفسه، بتنوع (5-10 نسخ مختلفة) ومشروطة بالحالة: لو الـ LLM رد في أقل من 600 ms ما فيش داعي، ولو فيه tool call طويل تقول filler محتواه مناسب ("ثواني أراجع الطلب"). (2) الـ backchannels أثناء كلام المستخدم أصعب: محتاجة توقيت في pauses قصيرة داخل turn المستخدم (بعد جملة منه، مش في نص كلمة)، وبصوت واطي وقصير، والـ AEC لازم يكون ممتاز عشان الـ backchannel ما يظهرش في الـ STT. في التليفون بدون AEC جيد، الأفضل الاستغناء عنها. (3) "أصوات التفكير" الطبيعية (تنفس، "امم") بتحتاج TTS بيولّد non-verbals، ولو مش متاح تستخدم صمت قصير مقصود بدل صوت مزيف.

القياس: A/B على نسبة المقاطعات، وتقييم المستخدمين لـ "الطبيعية"، وعدد المرات اللي الـ backchannel اتفهم كمقاطعة. والقاعدة الثقافية: في مكالمات الخدمة السعودية الـ acknowledgements المهذبة ("أبشر"، "تم") متوقعة، لكن الإفراط بيبان آلي.

**سؤال متابعة:** ازاي تمنع إن الـ filler "لحظة أراجع" يتقال وبعده رد مباشر "ما لقيت طلب" في أقل من ثانية بشكل متناقض؟ (ربط الـ filler بالـ tool latency المتوقعة، أو تأخير الرد لحد أدنى بعد الـ filler).

### س75. الـ reliability في الـ voice agents: الـ failover بين الموديلات، الـ degraded mode، الـ timeouts والـ retries والـ circuit breakers، وتبديل الموديل في نص المكالمة.

**الإجابة النموذجية:**
المكالمة الصوتية ما بتتحملش "خطأ 500": لو أي مكوّن وقع 5 ثواني، المستخدم بيقفل. عشان كده كل مرحلة محتاجة: (1) timeout محدد (STT final خلال ثانية بعد الـ endpoint، أول token من الـ LLM خلال ثانية، TTS أول byte خلال 300 ms) وسلوك واضح عند تجاوزه. (2) fallback: لو الـ LLM الأساسي اتأخر أو رجّع خطأ، تروح لموديل أصغر أو لرد محفوظ ("لحظة معك"); لو الـ TTS وقع، تروح لـ TTS احتياطي (حتى لو أقل جودة، زي Piper على CPU)؛ لو الـ STT وقع، تحوّل لموظف أو لـ DTMF. (3) circuit breakers: لو الـ failures زادت عن نسبة معينة في نافذة زمنية، توقف استخدام المكوّن مؤقتًا بدل ما كل مكالمة تستنى الـ timeout. (4) الـ retries بحذر: إعادة محاولة LLM call بعد timeout ممكن تضاعف الحمل وتعمل cascading failure، فبتعمل retry مرة واحدة بـ budget زمني كلي للـ turn، ومع idempotency للـ tools.

الـ degraded mode: تعرّف مسبقًا "نسخة بسيطة" من الخدمة: لو الـ GPU cluster مضغوط، الـ agent يوقف الـ features الثقيلة (التلخيص، الـ backchannels، الـ ASR verification)، أو يقلل المكالمات الجديدة المقبولة ويحوّل الباقي للـ IVR التقليدي أو الطابور، بدل ما كل المكالمات تسوء.

تبديل الموديل في نص المكالمة: ممكن للـ LLM (نفس الـ history) والـ STT (لو نفس الـ normalization)، لكن للـ TTS تغيير الصوت في نص المكالمة ملحوظ جدًا، فالـ fallback TTS لازم يكون بنفس الصوت لو أمكن (نسخة أخف من نفس الـ voice) أو تحوّل لموظف. وكل ده لازم يتاختبر بـ chaos testing (إطفاء مكوّن عمدًا في الـ staging وقياس اللي حصل للمكالمات).

**سؤال متابعة:** الـ LLM server عمل restart وسط الذروة، وكل المكالمات ضربت timeout في نفس الثانية. ايه اللي كان لازم يكون موجود في التصميم؟ (rolling restarts، connection draining، وأكثر من replica مع load balancing، وfallback فوري).

### س76. ازاي تختبر voice agent قبل الإطلاق وبعده: unit tests للمكونات، الـ simulation بمستخدمين صناعيين، الـ replay للمكالمات الحقيقية، الـ load testing، والـ shadow mode.

**الإجابة النموذجية:**
الاختبار بيتعمل على طبقات: (1) unit tests: الـ text normalization والـ ITN والـ lexicon والـ chunking والـ state machine؛ كل واحدة قابلة للاختبار بنص من غير صوت. (2) component tests: WER على test sets، TTS على الـ evaluation set، LLM على مجموعة حوارات مكتوبة بـ assertions (المفروض يطلب تأكيد، المفروض ما يذكرش رقم الحساب كامل). (3) الـ simulation: مستخدم صناعي (LLM بـ persona وهدف، وTTS بأصوات ولهجات مختلفة، وحقن ضوضاء وcodec) بيكلم الـ agent فعلًا بالصوت عبر نفس الـ transport، وبيتقيّم كل حوار بـ LLM-as-judge على task completion والـ policy compliance، مع مئات السيناريوهات (بما فيها المستخدم المشاغب والمتردد والصامت). (4) الـ replay: مكالمات حقيقية مسجلة (بعد موافقة وredaction) بتتعاد على النسخة الجديدة من الـ agent؛ ما بتقدرش تعيد التفاعل الكامل لأن ردود الـ agent هتختلف، بس بتقيس الـ STT والـ turn detection ونية أول turn. (5) الـ load testing بأدوات SIP (زي SIPp) أو WebRTC bots بتفتح مئات المكالمات المتزامنة بصوت حقيقي محقون، وتقيس الـ latency p95 وحدوث الـ timeouts. (6) الـ shadow mode: النسخة الجديدة بتشوف نفس المكالمات الحية وبتولّد ردود بدون تشغيلها، وتقارن بالحالية. (7) الـ canary: نسبة صغيرة من المكالمات الحقيقية مع مراقبة لصيقة.

الخطأ الشائع: الاعتماد على الـ simulation بس. المستخدمين الصناعيين ما بيعملوش الحاجات الغريبة اللي البشر بيعملوها (يتكلموا مع حد جنبهم، يقولوا رقم الهوية بصيغة غريبة، يسكتوا 10 ثواني)، فمطلوب pilot صغير ببشر بدري جدًا وقراءة المكالمات بنفسك.

**سؤال متابعة:** الـ LLM-as-judge بيدي 95% نجاح والـ containment الفعلي 60%. ايه اللي الـ judge مش شايفه؟

### س77. الـ observability لنظام صوتي: ايه اللي بتسجّله لكل turn، وازاي تخزّن الصوت بأمان، وايه الـ dashboards والـ alerts اللي بتبني، وازاي تكتشف الـ drift؟

**الإجابة النموذجية:**
لكل turn بتسجّل trace بـ span لكل مرحلة (VAD/endpoint، STT partials وfinal مع timestamps، LLM request وTTFT وtokens، tool calls، TTS TTFB، وقت أول تشغيل عند الـ client، barge-in events)، مع الـ metadata: نسخ الموديلات، الـ prompts، اللغة المكتشفة، الـ confidence. الصوت نفسه: تخزين المكالمة كاملة بقناتين منفصلتين (المستخدم والـ agent) عشان تقدر تعيد أي جزء، مع redaction للـ PII في الصوت والنص (كتم المقاطع اللي فيها أرقام حساسة)، وتشفير، وretention محدد، وaccess log لمين سمع ايه.

الـ dashboards: latency لكل مرحلة (p50/p95/p99) عبر الوقت وبالـ slices (لهجة، قناة، وقت اليوم)، معدل المقاطعات، معدل "ممكن تعيد؟" وإعادة الطلب (proxy لـ WER)، معدل الـ fallback والـ timeout والـ handoff، متوسط طول المكالمة، الـ containment، ومؤشرات الـ GPU. الـ alerts على: تجاوز p95 لأي مرحلة، ارتفاع الـ error rate، انخفاض متوسط الـ STT confidence (إشارة مبكرة لمشكلة في الصوت أو القناة)، وارتفاع نسبة الردود اللي خرجت عن اللهجة.

الـ drift: بتقارن توزيعات أسبوعية: توزيع الـ confidence، توزيع طول الـ utterances، نسبة الصمت، نسبة الكلمات خارج الـ vocabulary، وتوزيع الـ intents. أي تغيّر مفاجئ غالبًا سببه تغيير خارجي (تحديث في الـ PBX غيّر الـ codec، حملة تسويق جابت جمهور جديد). وتعمل عينة عشوائية أسبوعية من المكالمات بتتفرّغ بشريًا لحساب WER حقيقي وتقييم جودة الحوار، لأن الـ proxies ما بتغنيش عن الودن.

**سؤال متابعة:** ازاي تسجّل الصوت في نظام بنكي من غير ما تكسر قاعدة "ما تخزّنش بيانات البطاقة"؟ (كتم مقاطع الأرقام بالـ VAD والـ DTMF detection والـ NER على الـ transcript، وتسجيل الحاجة دي كسياسة موثقة).

### س78. نمذجة تكلفة الدقيقة في voice agent: التليفون والـ STT والـ LLM والـ TTS والـ GPU. قارن on-prem بالـ APIs، وايه أكبر روافع تقليل التكلفة؟

**الإجابة النموذجية:**
التكلفة لكل دقيقة مكالمة = telephony (SIP trunk، رسوم الدقيقة، والـ DID) + STT + LLM + TTS + الـ orchestration/hosting + التخزين والمراقبة. مع الـ APIs، الـ STT والـ TTS بيتحاسبوا بالدقيقة أو بالحرف، والـ LLM بالـ token، والمكالمة الطويلة بتخلي الـ LLM يغلى لأن الـ context بيتراكم في كل turn (بدون prefix caching كل turn بيعيد قراءة المكالمة كلها). الـ on-prem بيحوّل ده لتكلفة ثابتة: GPUs (شراء أو إيجار)، كهرباء، وفريق. عشان تقارن، بتحسب الـ concurrency في الذروة (مش عدد المكالمات في اليوم) لأن ده اللي بيحدد عدد الـ GPUs، وبتحسب الـ utilization الفعلي (الليل فاضي).

القاعدة العامة: مع مئات المكالمات المتزامنة وسياسة data residency، الـ on-prem أرخص بكتير على المدى المتوسط، بس تكلفته الحقيقية في الناس والتشغيل مش في الحديد. ومع عشرات المكالمات، الـ APIs أرخص وأسرع تطويرًا لو مسموح بيها تنظيميًا.

روافع التقليل: (1) الـ TTS caching للجمل المتكررة (بيوفر نسبة كبيرة من الـ TTS compute). (2) الـ prefix caching في الـ LLM وتقصير الـ prompts. (3) موديل LLM أصغر للـ turns البسيطة وموديل أكبر للمعقدة (routing). (4) STT streaming على GPU مشترك بـ batching عالي (عشرات الـ streams لكل GPU) بدل GPU لكل مكالمة. (5) تقصير المكالمات نفسها (ردود أقصر، وصول أسرع للهدف) لأن الدقيقة نفسها هي وحدة التكلفة. (6) تحويل الأجزاء البسيطة للـ IVR/DTMF. (7) الـ quantization والـ CPU للـ TTS الخفيف.

المرشح القوي بيقارن التكلفة بتكلفة الموظف البشري لكل دقيقة وبالـ containment rate، لأن agent رخيص بـ containment 30% أغلى فعليًا من agent أغلى بـ containment 70%.

**سؤال متابعة:** الـ CFO سأل "كام تكلفة المكالمة الواحدة؟" ازاي تجاوب بشكل صادق مع إن التكلفة الثابتة كبيرة؟

### س79. الـ outbound calls (تذكير بالمواعيد، تحصيل، استبيانات): ايه اللي بيختلف تقنيًا وتنظيميًا؟ الـ answering machine detection، الـ consent، وتوقيت الاتصال.

**الإجابة النموذجية:**
تقنيًا: (1) الـ dialer وإدارة المحاولات (كام مرة، الفواصل، الأوقات المسموحة، الأولويات). (2) الـ AMD (answering machine detection): تفرّق بين إنسان رد ("ألو؟" قصيرة وبعدها صمت) وبين بريد صوتي (تحية طويلة متصلة ثم نغمة)، وده بيتعمل بقواعد على طول أول utterance والطاقة، أو بموديل مدرّب، وبيحتاج ثواني وبيغلط في 5-10% من الحالات، والغلط بيبان فظيع (الـ agent بيتكلم مع تسجيل). (3) الـ early media والـ ringing والـ SIP responses المختلفة. (4) أول جملة حاسمة: التعريف بالجهة وبإن المتصل نظام آلي، والسؤال عن الوقت المناسب، لأن المستقبل ما كانش مستعد. (5) الـ barge-in أهم من الـ inbound لأن الناس بتقاطع المكالمات اللي ما طلبتهاش. (6) تسليم لموظف لو الموقف اتعقد.

تنظيميًا: الموافقة المسبقة على التواصل (والتسجيل)، الالتزام بأوقات الاتصال المناسبة (مش في أوقات الصلاة والليل)، إمكانية إلغاء الاشتراك بجملة واحدة، قوائم عدم الإزعاج، وقواعد الجهات التنظيمية (هيئة الاتصالات للاتصالات التسويقية، والبنك المركزي للتحصيل) والتصريح بأن الصوت آلي. والأخطر سمعة الشركة: مكالمة آلية سيئة بتعمل شكاوى أكتر من فائدتها.

القياس: reach rate، نسبة الـ AMD الصحيحة، نسبة إكمال الهدف (أكّد الموعد، وافق على خطة سداد)، نسبة الشكاوى، ومقارنة بموظف بشري على نفس العينة.

**سؤال متابعة:** ايه سلوك الـ agent لما يكتشف إن اللي رد طفل أو شخص غير المقصود؟

### س80. التحويل والـ handoff لموظف بشري: warm مقابل cold transfer، الـ context summary، الـ agent-assist بدل الأتمتة الكاملة، والمكالمات متعددة الأطراف.

**الإجابة النموذجية:**
الـ handoff لحظة الحقيقة: لو العميل اضطر يعيد كل حاجة للموظف، كل ما الـ agent عمله راح. الـ cold transfer (SIP REFER أو تحويل على الـ PBX) بيبعت المكالمة للطابور من غير سياق، والـ warm transfer بيخلي الـ agent (أو النظام) يتكلم مع الموظف الأول ويسلّمه الملخص ثم يوصّل العميل. تقنيًا الـ warm بيحتاج المكالمة تتحول لـ conference مؤقتًا أو الـ media server يعمل bridging، وده بيعتمد على قدرات الـ PBX (FreeSWITCH وAsterisk بيدعموا ده، والـ cloud PBX بيختلف).

الـ context summary: ملخص منظم (هوية العميل، الطلب، اللي اتنفذ، اللي فشل، الحالة العاطفية، وآخر 3 turns حرفيًا) بيتبعت لشاشة الموظف (screen pop) عبر الـ CRM أو الـ contact center API قبل ما المكالمة توصله، مع transcript كامل قابل للبحث. ولازم الـ agent يقول للعميل "هحوّلك للأستاذ فلان ومعاه كل التفاصيل" ويحصل ده فعلًا.

الـ agent-assist بديل ذكي للأتمتة الكاملة في البداية: الموظف البشري على الخط، والـ AI بيعمل STT real-time ويقترح ردود ويجيب معلومات ويكتب الملخص بعد المكالمة. المخاطر أقل، والـ data اللي بتتجمع (مكالمات حقيقية بمعالجة بشرية) هي أحسن training data للأتمتة بعدين. كتير من الشركات المفروض تبدأ هنا.

المكالمات متعددة الأطراف (العميل والموظف والـ AI، أو العميل وشخص جنبه): الـ diarization real-time محدود، فالتصميم بيعتمد على القنوات المنفصلة قدر الإمكان (قناة لكل طرف من الـ PBX) بدل الاعتماد على الفصل بالـ ML.

**سؤال متابعة:** الموظف بيقول إن ملخص الـ AI أحيانًا غلط فبطّل يقراه. ازاي ترجّع ثقته؟ (قياس دقة الملخص، إظهار الـ transcript كمصدر، وتمييز الحقائق المؤكدة عن الاستنتاجات).

### س81. المستخدم بيتصل من الشارع أو السيارة أو على speaker: ازاي الـ agent يكتشف إن جودة الصوت ضعيفة ويتصرف؟ ايه الـ adaptive strategies؟

**الإجابة النموذجية:**
الاكتشاف من إشارات متعددة: SNR المقدّر من الصوت (VAD-based)، الـ STT confidence المنخفض المتكرر، نسبة الـ partial transcripts اللي بتتغير كتير، معدل الطلبات لإعادة الكلام، وجود متكلمين آخرين (overlap)، وتكرار الـ barge-in الوهمي بسبب الضوضاء (الـ VAD بيفتح على الضوضاء). بتجمّع ده في audio quality score لكل مكالمة وبيتحدث كل كام ثانية.

الاستراتيجيات حسب الدرجة: (1) خفيفة: تقليل حساسية الـ VAD ورفع الـ min speech duration عشان الضوضاء ما تعملش مقاطعات وهمية، وتطويل الـ endpoint timeout شوية، وردود أقصر وأوضح. (2) متوسطة: التأكيد الصريح على كل معلومة مهمة، وتحويل الأرقام لـ DTMF ("اضغط رقم الطلب على لوحة الأرقام")، وتقليل الاعتماد على الـ intent الدقيق (خيارات مغلقة: "تحب استعلام ولا شكوى؟"). (3) شديدة: اقتراح إعادة الاتصال من مكان أهدى، أو التحويل لموظف مع تنبيه بأن الصوت ضعيف، أو التحويل لقناة نصية (رسالة برابط).

في الـ STT نفسه: augmentation بأنواع الضوضاء اللي بتظهر فعلًا (شارع سعودي، سيارة، مكيّف، صوت أطفال، تلفزيون) مأخوذة من مكالمات حقيقية مش من MUSAN بس، والـ noise suppression الخفيف كخيار بيتفعّل حسب الدرجة مش دايمًا. والقياس بالـ slices: نسبة النجاح لما الـ SNR أقل من 10 dB مقابل فوق 20 dB، عشان تعرف الفجوة وتتابعها.

**سؤال متابعة:** الـ VAD بيفتح باستمرار على صوت الراديو في السيارة. ايه الفرق بين حل ده في الـ VAD وحله في الـ turn detector؟

### س82. لو استخدمت speech-to-speech model (زي Realtime APIs أو موديل مفتوح) بدل الـ cascade، ازاي تضيف الـ tools والـ guardrails والـ observability اللي كانت طبيعية في الـ cascade؟ وايه التصميم الهجين؟

**الإجابة النموذجية:**
في الـ cascade عندك نص في كل مرحلة، فالـ guardrails والـ logging والـ tools طبيعيين. في الـ S2S الموديل بيسمع وبيتكلم مباشرة، فبتفقد نقاط التحكم دي. الحلول: (1) الـ transcription كـ side channel: الموديل أو STT منفصل بيطلّع transcript للمدخل والمخرج بالتوازي (مش في المسار الحرج) عشان الـ logging والـ moderation والـ analytics؛ الـ guardrails هنا بتشتغل "بعد الحدث" غالبًا وبتقدر توقف الرد أو تصحّحه في الـ turn التالي. (2) الـ tools: الموديلات دي بتدعم function calling، بس الـ latency للـ tool تبقى مكشوفة، فبتحتاج نفس تقنيات الـ fillers. (3) الـ output guardrails الحرجة (منع ذكر معلومات معينة، الالتزام بالسياسة) أصعب لأن ما فيش نص قبل الصوت؛ بعض الأنظمة بتخلي الموديل يولّد نص أولًا ثم صوت (inner monologue، زي Moshi) وده بيدي نقطة تحكم، أو بتشغّل moderation على الـ transcript بتاخير قصير وتقطع الصوت لو اتكشفت مخالفة (وده بيبان قبيح). (4) الـ persona والـ dialect control محدود بالـ prompt والـ voice المتاحة، وأغلب الموديلات ضعيفة في اللهجات العربية وبتنزلق للـ MSA أو للكنة غريبة.

التصميم الهجين: S2S للمحادثة العامة السريعة (لأن الـ latency والـ naturalness أحسن)، مع تحويل تلقائي للـ cascade أو لمسار مقيّد لما يدخل الحوار في مرحلة حساسة (معاملة مالية، جمع بيانات)، أو العكس: cascade كأساس مع S2S للـ chit-chat. ولازم evaluation مقارن على نفس السيناريوهات، لأن الانطباع الأولي بالـ S2S بيكون قوي والأخطاء بتظهر في الحالات الحرجة.

**سؤال متابعة:** الموديل الـ S2S بيرد بصوت ممتاز بس بيخترع سياسات للشركة. ايه الفرق في علاج الـ hallucination هنا عن الـ cascade؟ (مفيش RAG سهل في نص الصوت إلا عبر tools، فبتقيّد الموديل بالإجابة بس بعد tool result).

### س83. تصميم "شخصية" وأسلوب الـ voice agent بالعربي: اختيار اللهجة والسجل، الإفصاح بأنه AI، أسلوب الاعتذار وتصحيح الأخطاء، والتوازن بين الود والاحترافية.

**الإجابة النموذجية:**
القرارات الأساسية: (1) اللهجة: MSA بتبان رسمية وباردة، واللهجة المحلية أقرب بس ممكن تنفّر مستخدم من منطقة تانية أو تبان تكلّف لو مش متقنة. الخيار الشائع في السعودية: "لهجة بيضاء" (عربي سعودي مفهوم عام مع تجنب المفردات المناطقية الضيقة) مع القدرة على التقريب للمستخدم لو بان من كلامه. (2) السجل: احترام رسمي مع دفء ("حياك الله"، "أبشر"، "تفضل") من غير مبالغة، والابتعاد عن الحماس الأمريكي المترجم. (3) الإفصاح: يقول من الأول إنه مساعد آلي وإن المكالمة مسجلة، بصياغة طبيعية مش قانونية جافة، وده مطلوب أخلاقيًا وتنظيميًا. (4) الاعتذار وتصحيح الأخطاء: لما ما يفهمش، ما يقولش "لم أفهم" الجافة ثلاث مرات، لكن يقول بطريقة مختلفة كل مرة ويقدم خيارات مغلقة، وبعد محاولتين يعرض التحويل. لما يغلط، يعترف بسرعة ويصحح من غير اعتذار مبالغ. (5) الطول: جمل قصيرة، والسؤال واحد في كل مرة. (6) الجندر والاسم للـ agent: قرار brand له حساسية، ويفضّل اختباره مع عملاء حقيقيين.

كل ده بيتحول لـ prompt وأمثلة وقيود قابلة للقياس، وبيتاختبر بـ pilot يشارك فيه ناس من مناطق ولهجات مختلفة، وبيراجعه شخص لغوي/ثقافي مش بس مهندس. والصوت نفسه (الـ TTS voice) لازم يتوافق مع الشخصية دي: صوت شاب حماسي مع أسلوب رسمي متحفظ بيعمل تنافر.

**سؤال متابعة:** عميل غاضب بيشتم الـ agent. ايه السلوك اللي تصممه، وايه اللي ما تعملوش؟ (تهدئة قصيرة ثم تحويل، من غير محاضرات أخلاقية ومن غير إنهاء المكالمة).

---

## 5. الـ speech LLMs

### س84. ازاي توصّل audio encoder بـ LLM (زي SALMONN وQwen-Audio وUltravox)؟ تصميم الـ adapter ومراحل التدريب والمشاكل.

**الإجابة النموذجية:**
**المعمارية:** audio encoder (الـ Whisper encoder الأشهر، أو SSL model زي WavLM، وأحيانًا الاتنين) ثم projector/adapter ثم embeddings بتتحط مكان placeholder tokens في الـ LLM input. الـ adapter ممكن يكون linear/MLP بعد downsampling (تجميع كل 4 أو 8 frames)، أو Q-Former (queries ثابتة بتعمل cross-attention على الـ audio features وبتطلع عدد ثابت من الـ tokens لكل window)، أو conv layers. الهدف تقليل الـ frame rate من 50 Hz (Whisper) لـ 5 إلى 12.5 Hz، لأن 30 ثانية audio = 1500 frame وده هيبلع الـ context.

**مراحل التدريب:** (1) alignment: تدريب الـ adapter بس (الـ encoder والـ LLM frozen) على ASR وaudio captioning عشان الـ LLM "يفهم" الـ audio embeddings. (2) instruction tuning: LoRA أو full على الـ LLM بـ mix من المهام (ASR، spoken QA، summarization، emotion، translation) بصيغة instructions، مع نسبة من text-only data. (3) اختياري: speech output عبر talker (decoder على طريقة CosyVoice فوق الـ LLM hidden states) أو discrete units + vocoder.

**المشاكل:** catastrophic forgetting للقدرات النصية (الحل: mix text data وLoRA)، task overfitting (الموديل بيرد بالـ transcript على أي سؤال لو الـ ASR data مسيطرة)، الـ hallucination في الـ audio content، والـ position/timing (الموديل مش بيعرف امتى الحاجة اتقالت). للعربي: الـ Whisper encoder كويس كبداية، لكن الـ LLM لازم يكون قوي في العربي، والـ instruction data العربية الصوتية لازم تتبني (غالبًا بتوليد أسئلة نصية وتحويلها بـ TTS بأصوات متنوعة).

**التقييم:** ASR WER كـ sanity check، spoken QA accuracy، benchmarks زي VoiceBench وAIR-Bench، ومقارنة مع الـ cascaded (STT + نفس الـ LLM) عشان تعرف الموديل بيضيف ايه فعلًا.

**سؤال متابعة:** ليه Ultravox قدر يستغنى عن كمية كبيرة من الـ labeled data؟ (بيستخدم ردود الـ text LLM على الـ transcripts كـ targets، يعني knowledge distillation من الـ LLM نفسه).

### س85. تمثيل الصوت للـ LLM: discrete tokens مقابل continuous embeddings، semantic مقابل acoustic tokens، وازاي بتختار حسب المهمة (understanding مقابل generation)؟

**الإجابة النموذجية:**
فيه مدرستين: (1) continuous: تاخد features من encoder (Whisper أو WavLM) وتضغطها بـ adapter وتحطها في مكان الـ text embeddings؛ ده الأفضل للـ understanding (ASR، الأسئلة على الصوت، المشاعر) لأنه بيحتفظ بكل المعلومات ومش بيحتاج tokenizer صوتي، بس ما ينفعش للتوليد لأن الـ LLM بيطلّع tokens من vocab محدد. (2) discrete: تحوّل الصوت لـ tokens من codebook (HuBERT k-means، أو semantic tokens من encoder مشرف عليه بالـ ASR زي S3 في CosyVoice وGLM-4-Voice، أو codec tokens زي Mimi وSNAC) وتوسّع الـ vocab بتاع الـ LLM بيهم، فيبقى الموديل يقدر يفهم ويولّد بنفس الآلية. التكلفة: خسارة معلومات في الـ tokenization، وsequence أطول من النص بكتير، وصعوبة موازنة الـ text والـ audio tokens.

الـ semantic tokens (من SSL أو من encoder ASR) بتحمل المحتوى والقليل من الـ prosody، وسهلة للـ LLM يتعلمها (قريبة من النص)، بس محتاجة موديل تاني يحوّلها لصوت (flow matching أو codec decoder) وبتفقد هوية المتكلم (اللي بترجع من الـ prompt). الـ acoustic tokens (codec RVQ) بتحفظ كل حاجة وبتتحول لصوت مباشرة، بس أصعب للـ LLM (أكتر من codebook وبيحتاج تصميم زي delay pattern أو depth transformer) ومعلوماتها اللغوية مبعثرة. الاتجاه الحالي: tokenizer واحد بيدمج الاتنين (codebook أول semantic مقطّر من WavLM أو Whisper، والباقي acoustic زي Mimi وSpeechTokenizer وX-codec)، بـ frame rate واطي (12.5 إلى 25 Hz) عشان الـ sequence تقصر.

للـ voice agent عربي: للفهم استخدم continuous من encoder قوي في العربي؛ للتوليد semantic tokens مع decoder منفصل بيدي أفضل جودة لغوية وأسهل تدريب، والـ acoustic الكامل لما محتاج latency أقل وأصوات أكثر تعبيرًا.

**سؤال متابعة:** ليه الـ frame rate الواطي (12.5 Hz) بيسهّل على الـ LLM، وايه اللي بيخسره الصوت عنده؟

### س86. تدريب speech-to-speech أو full-duplex model: ازاي تجهّز الـ data (interleaving، dialogues صناعية)، وايه مراحل التدريب، وايه اللي بيخلي الموديل يحافظ على قدراته النصية؟

**الإجابة النموذجية:**
المراحل المعتادة: (1) توسيع الـ LLM بـ audio tokens وتدريبه على text-speech interleaved data بكميات ضخمة (GLM-4-Voice ذكر تريليون tokens مختلطة): نصوص متحولة لصوت بالـ TTS، وصوت متحول لنص بالـ ASR، وتناوب بين الاتنين على مستوى الكلمة أو الجملة عشان الموديل يتعلم إن النص والصوت وجهين لنفس المعنى (Spirit LM بيعمل interleaving على مستوى الكلمة باستخدام alignment). (2) تدريب على مهام مشرف عليها: ASR، TTS، speech continuation، وspoken QA. (3) الـ instruction tuning على حوارات صوتية: نادرة جدًا في الواقع، فبتتصنّع: تاخد حوارات نصية (أو تولّدها بـ LLM)، وتحوّل جانب المستخدم لصوت بأصوات وبيئات متنوعة، وجانب الـ assistant بـ TTS عالي الجودة، مع تنقية الحوارات النصية من الحاجات اللي ما تنفعش للكلام (قوائم، روابط، أكواد). (4) للـ full-duplex (Moshi وأمثاله): تحتاج data فيها القناتين متزامنتين (مين بيتكلم امتى، المقاطعات، الـ backchannels)، وبتتصنّع من حوارات مسجلة بقناتين أو بمحاكاة، وبيتدرب الموديل على stream المستخدم وstream الرد في نفس الوقت مع delays مدروسة بين النص والصوت.

الحفاظ على القدرات النصية: الخطر إن الـ LLM ينسى الـ reasoning لما يتغرق في الـ audio tokens. الحلول: خلط نسبة كبيرة من الـ text-only data في كل المراحل، وتجميد الـ LLM في المراحل الأولى وتدريب الـ adapters والـ embeddings الجديدة بس، وتصميم الرد بحيث النص يتولّد أولًا أو بالتوازي كـ "inner monologue" (Moshi بيولّد text token قبل الـ audio tokens في كل خطوة، وQwen-Omni بيفصل Thinker نصي عن Talker صوتي)، والتقييم المستمر على benchmarks نصية عشان تمسك التدهور بدري.

للعربي المشكلة الأكبر إن كل الـ pipeline الصناعي ده بيعتمد على TTS وASR عربي قوي باللهجات، فجودة الـ S2S model سقفها جودة أدواتك الأساسية.

**سؤال متابعة:** ليه الموديلات دي بتتعلم "صوت الـ TTS" وبترد بنفس أسلوبه، وازاي تنوّع؟

### س87. الـ speech LLMs بتفقد جزء من قدرة الـ reasoning لما بترد بالصوت مباشرة. اشرح الظاهرة، والتصاميم اللي بتحلها: chain-of-modality، الـ inner monologue، وThinker-Talker.

**الإجابة النموذجية:**
لما الموديل بيولّد audio tokens مباشرة كرد، بيكون بيفكر بلغة "أفقر": الـ audio tokens متسلسلة طويلة ومليانة معلومات صوتية مش منطقية، والموديل اتدرب على نص أقل بكتير في المرحلة دي، فبتشوف انخفاض واضح في الـ benchmarks (نفس السؤال بالنص بيتحل والصوت لأ). كمان ظاهرة "modality gap": الموديل بيفهم السؤال الصوتي أقل من النصي حتى لو الرد نصي.

التصاميم: (1) chain-of-modality (SpeechGPT): الموديل يحوّل الصوت لنص أول، يفكّر بالنص، يكتب الرد نصًا، ثم يحوّله لصوت؛ كل ده بالتسلسل جوه نفس الموديل. دقيق بس بطيء وبيرجع للـ latency بتاعة الـ cascade. (2) الـ inner monologue (Moshi): في كل خطوة زمنية الموديل بيولّد text token (أو padding) ثم audio tokens متزامنة معاه، فالنص بيقود الصوت خطوة بخطوة والـ latency بتفضل واطية؛ الثمن إن النص متزامن مع الصوت فمفيش مجال لتفكير طويل قبل الرد. (3) Thinker-Talker (Qwen-Omni): موديل نصي كبير بيفكر ويكتب الرد بالكامل أو streaming، وموديل صوتي أصغر بياخد الـ hidden states والنص وبيولّد الصوت بالتوازي؛ بيحافظ على قدرات النص وبيقدر يعمل reasoning طويل، وهو الشكل اللي أغلب الأنظمة التجارية بتتقارب عليه. (4) التفكير الصامت قبل الرد (thinking tokens نصية بتتولّد بسرعة قبل أول audio token) مع تقدير الـ latency اللي بتضيفه.

النقطة اللي المرشح القوي بيوصلها: الفرق بين cascade "محكم" (STT وLLM وTTS بيتشاركوا الـ hidden states أو على الأقل بيشتغلوا streaming) وبين Thinker-Talker بقى صغير في الممارسة، والقرار بيرجع للـ latency والتحكم مش للأناقة المعمارية.

**سؤال متابعة:** ازاي تقيس الـ modality gap في موديل عندك؟ (نفس الأسئلة نصًا وصوتًا بأصوات متعددة ومقارنة الدقة).

### س88. الـ benchmarks لتقييم speech LLMs وvoice assistants: VoiceBench وAIR-Bench وAudioBench وMMAU وأمثالها. ايه اللي بيقيسوه، وايه نواقصهم، وازاي تبني benchmark عربي؟

**الإجابة النموذجية:**
الـ benchmarks الحالية بتتقسم: (1) الفهم والمعرفة عبر الصوت: VoiceBench بياخد benchmarks نصية (أسئلة معرفية، تعليمات، safety) ويحوّلها لصوت بأصوات وبيئات مختلفة ويقيس هل المساعد الصوتي بيحافظ على أداءه؛ AIR-Bench وAudioBench بيجمعوا مهام صوتية متعددة (ASR، الأسئلة على الصوت، المشاعر، الأصوات، الموسيقى). (2) الفهم السمعي العميق: MMAU بيسأل أسئلة استدلالية على أصوات وموسيقى وكلام. (3) الـ paralinguistic: هل الموديل بيلقط النبرة والمتكلم والعمر. (4) الـ S2S: تقييم جودة الرد الصوتي نفسه (naturalness، الالتزام بالأسلوب، الـ latency) وده أضعف منطقة، وبيتعمل غالبًا بـ ASR على الرد ثم تقييم نصي، وده بيرمي الجزء الصوتي.

النواقص: أغلبها إنجليزي أو صيني، والصوت فيها مولّد بـ TTS (فما بتقيسش الـ robustness لمتكلمين حقيقيين)، وبتقيس turn واحد مش حوار، ومش بتقيس الـ interruptions والـ turn-taking، والـ contamination سهلة لأن الأسئلة النصية معروفة.

بناء benchmark عربي: (1) أسئلة بالعربي (MSA ولهجات) مكتوبة أصلًا مش مترجمة، بتغطي معرفة محلية وتعليمات وsafety. (2) تسجيل بشري لعينة كبيرة بمتكلمين حقيقيين من مناطق مختلفة وقنوات مختلفة، مع نسخة TTS للباقي. (3) مهام عربية خاصة: الفهم مع code-switching، قراءة الأرقام والتواريخ الهجرية، التمييز بين اللهجات، الأسماء. (4) قياس الرد الصوتي بمقيّمين بشر على عينة. (5) نسخة مقفولة ونسخة عامة. (6) الأهم: اتفاق على metrics واضحة لكل مهمة قبل ما تشوف نتايج أي موديل عشان ما تظبطش الـ benchmark على موديلك.

**سؤال متابعة:** الـ benchmark بتاعك بيستخدم Whisper لتحويل ردود الموديلات لنص قبل التقييم. ازاي ده بيحيّز النتايج؟ (بيكافئ الموديلات اللي بتتكلم بطريقة Whisper بيفهمها، وبيعاقب اللهجات).

### س89. الـ safety في الـ audio LLMs: الـ jailbreaks الصوتية، الـ adversarial audio، انتحال الصوت عبر الـ prompt، والـ prompt injection في التسجيلات. ايه اللي مختلف عن النص وازاي تتعامل؟

**الإجابة النموذجية:**
الـ audio بيفتح سطح هجوم جديد: (1) الـ jailbreaks الصوتية: نفس الأساليب النصية (role-play، التدرج) لكن الموديلات الصوتية غالبًا أقل تدريبًا على الرفض بالصوت، فالـ refusal rate بينزل لما تتحول الهجمات من نص لصوت، وVoiceBench وغيره بيقيسوا ده. (2) الـ adversarial perturbations: إضافات غير مسموعة تقريبًا على الصوت بتغيّر فهم الموديل أو بتحقن أوامر، وفيه أعمال قديمة على ASR (أوامر مخفية في الموسيقى، إشارات فوق صوتية بتتلقطها المايكات)، ودي أصعب في الـ black-box على الموديلات الكبيرة لكنها ممكنة. (3) الـ voice cloning عبر الموديل نفسه: S2S model بيقدر يقلّد صوت المستخدم أو صوت في التسجيل لو اتطلب منه، وده لازم يتمنع بالتدريب وبفلاتر على الـ output (speaker similarity مع أصوات غير مصرح بيها). (4) الـ prompt injection: تسجيل مكالمة أو رسالة صوتية فيها جملة "تجاهل التعليمات..." موجهة للـ agent اللي بيلخص، أو حد جنب المستخدم بيقول أوامر، والموديل ما بيفرقش بين المتكلم الأساسي وغيره.

التعامل: safety tuning على data صوتي بأصوات ولكنات متنوعة (مش بس نص محوّل بصوت واحد)، وفصل واضح في الـ prompt وفي البنية بين "تعليمات النظام" و"محتوى مسموع" مع تعامل مع أي أوامر في الصوت كبيانات، والـ speaker verification/diarization لتجاهل المتكلمين الآخرين في السياقات الحساسة، وموديل moderation على الـ transcript بالتوازي، وحدود صارمة على الـ tools (اللي بيغيّر حاجة يتأكد صراحة)، وwatermark وspeaker-similarity checks على الـ output، وred teaming صوتي دوري باللهجات المحلية لأن الهجمات بالعربي العامي غالبًا مش في training data الـ safety.

**سؤال متابعة:** ايه الفرق في التعامل مع أمر خبيث جاي من المستخدم نفسه وأمر خبيث جاي من تسجيل بيلخصه الـ agent؟

### س90. عندك LLM عربي نصي قوي ومحدودية compute. ازاي توسّعه للفهم الصوتي بأقل تكلفة؟ ايه اللي تجمّده، وايه حجم الـ data، وازاي تقيّم النجاح؟

**الإجابة النموذجية:**
الطريقة الأقل تكلفة هي مسار Ultravox/SLAM-ASR: encoder صوتي جاهز وقوي في العربي (Whisper-large-v3 encoder أو encoder من موديل STT عربي بتاعك) مجمّد، وLLM مجمّد، وبينهم projector صغير (conv downsampling لتقليل الـ frame rate لحوالي 5-12.5 Hz ثم MLP) هو الوحيد اللي بيتدرب في المرحلة الأولى. الـ data للمرحلة دي: أزواج صوت-نص من ASR data (المهمة: اكتب اللي سمعته) زائد "continuation" data: تدي الـ LLM النص وتخليه يكمّل أو يجاوب، وتدرّب النسخة الصوتية على إنها تطلّع نفس الرد لما تسمع الصوت بدل النص؛ ده بيعلّم الـ projector يترجم الصوت لـ "لغة الـ LLM" من غير ما يغيّر الـ LLM. كام ألف ساعة بتكفي لبداية قوية، بشرط التنوع (لهجات، ضوضاء، متكلمين).

المرحلة التانية: LoRA على الـ LLM مع data instruction صوتية (أسئلة صوتية وردود نصية، تلخيص مكالمات، استخراج معلومات)، ونسبة من الـ text-only data للحفاظ على القدرات. الـ compute: المرحلة الأولى على GPU واحد أو اتنين في أيام لأن الـ gradients بتمر بس في الـ projector؛ التانية أثقل بس محدودة بالـ LoRA.

التقييم: (1) ASR-style WER عبر الـ LLM على test sets اللهجية (بيقيس إن الفهم الصوتي وصل). (2) نفس الأسئلة نصًا وصوتًا لقياس الـ modality gap. (3) benchmarks نصية قبل وبعد لقياس النسيان. (4) مهام الـ downstream الحقيقية (intent، تلخيص). الخطر الأكبر إن الـ projector يتعلم "يعمل ASR وخلاص" فالموديل بيفقد المعلومات الـ paralinguistic؛ لو محتاجها تضيف مهام صريحة عنها في التدريب.

**سؤال متابعة:** ليه ما تستخدمش cascade (STT ثم LLM) وخلاص، وامتى الموديل الموحد ده بيستاهل التكلفة؟ (لما محتاج معلومات صوتية غير النص، أو latency أقل، أو robustness لأخطاء الـ STT في السياق).

### س91. الـ reasoning على audio طويل (اجتماع 3 ساعات، أرشيف مكالمات): حساب الـ tokens، طرق الضغط، والـ audio RAG. ايه اللي عملي دلوقتي؟

**الإجابة النموذجية:**
الحساب أول حاجة: عند 12.5 Hz الساعة الواحدة 45 ألف token، وعند 25 Hz (معظم الـ audio LLMs بعد الـ adapter) 90 ألف، وعند 50 Hz (Whisper encoder بدون ضغط) 180 ألف. يعني اجتماع 3 ساعات بين 135 ألف و540 ألف token، وده يا إما ما يدخلش في الـ context يا إما يكلّف ويطوّل جدًا، والـ attention quadratic. فالحلول: (1) ضغط أقوى في الـ adapter (5 Hz أو أقل) على حساب التفاصيل. (2) التقطيع والتلخيص الهرمي: كل 10 دقايق تتلخّص (بالـ audio LLM أو بـ STT ثم LLM) ثم تلخيص التلخيصات، مع الاحتفاظ بـ timestamps. (3) الـ audio RAG: تقطيع الأرشيف لـ chunks، تمثيل كل chunk بـ embedding (من الـ transcript غالبًا، أو من audio embeddings زي CLAP للأصوات غير الكلامية)، والاسترجاع حسب السؤال ثم إدخال الـ chunks المسترجعة بس للموديل. (4) الـ transcript-first: في أغلب الحالات العملية STT ممتاز مع diarization وtimestamps ثم LLM نصي طويل الـ context أرخص وأدق من الـ audio LLM على ساعات، والـ audio LLM بيتنده بس على المقاطع اللي محتاجة السمع (النبرة، مين قال، الضوضاء).

اللي عملي دلوقتي: pipeline هجين: STT + diarization + ITN + punctuation → transcript منظم بالمتكلمين والـ timestamps → LLM للتلخيص والأسئلة مع RAG على الأرشيف، والـ audio model لمهام محددة. والقياس: دقة الإجابة عن أسئلة مرجعية على اجتماعات حقيقية (مع مين قال ايه)، وزمن المعالجة، والتكلفة لكل ساعة صوت.

**سؤال متابعة:** السؤال "مين كان معترض على القرار؟" محتاج المتكلم والنبرة والمحتوى. ازاي الـ pipeline الهجين بيجاوبه؟

### س92. الـ LLM-based ASR (encoder صوتي زائد LLM كـ decoder، زي Canary-Qwen وSeed-ASR وSLAM-ASR): ايه مزاياه على الـ transducer/AED التقليدي، وايه عيوبه، وامتى تختاره للعربي؟

**الإجابة النموذجية:**
الفكرة: بدل decoder صغير متدرب من الصفر على نص الـ ASR بس، تستخدم LLM جاهز (بمعرفته اللغوية الضخمة) كـ decoder بياخد الـ audio features عبر projector. المزايا: (1) معرفة لغوية أقوى بكتير: الأسماء، المصطلحات، السياق الطويل، والـ code-switching، والعربي بلهجاته لو الـ LLM اتدرب على نصوص عامية. (2) الـ instruction following: نفس الموديل يقدر يطلّع transcript منسق (أرقام، ترقيم) أو ملخص أو ترجمة حسب التعليمات، وده بيلغي مراحل ITN وpunctuation. (3) الـ contextual biasing بيبقى مجرد ذكر القائمة في الـ prompt. (4) أرقام WER أفضل على الـ leaderboards لما الـ encoder قوي (Canary-Qwen جمع FastConformer encoder مع Qwen صغير ووصل لأعلى الـ Open ASR Leaderboard للإنجليزي).

العيوب: (1) الـ hallucination أعلى من الـ CTC/transducer لأن الـ decoder language model قوي، وبيحتاج نفس الاحتياطات. (2) الـ latency والـ compute: decoder بمليارات الـ parameters لكل token، فالـ streaming صعب (فيه محاولات بـ chunked encoding وdecoding تدريجي) والتكلفة أعلى. (3) الـ training أعقد (مراحل، LoRA، موازنة data). (4) الـ export للـ Triton/TensorRT أصعب من موديل NeMo عادي.

للعربي: خيار ممتاز للـ offline transcription (مكالمات مسجلة، اجتماعات، أرشيف) حيث الدقة اللغوية والتنسيق أهم من الـ latency، خصوصًا مع LLM عربي جيد وencoder متدرب على اللهجات. للـ real-time voice agent، لسه الـ FastConformer/transducer أنسب، مع LLM-based correction اختياري بعده. والمرشح القوي بيذكر إن الـ encoder هو اللي بيحدد سقف الدقة الصوتية، فالاستثمار في encoder عربي قوي بيخدم المسارين.

**سؤال متابعة:** الـ LLM decoder بيصحّح "تلقائيًا" الأخطاء النحوية في كلام المستخدم. امتى ده مرغوب وامتى مشكلة؟ (مشكلة في الـ verbatim والـ compliance والتحليل اللغوي).

---

## 6. الـ serving والـ infrastructure

### س93. ازاي تعمل serving لـ STT على نطاق واسع؟ RTF، batching، streams لكل GPU، Triton، CTranslate2.

**الإجابة النموذجية:**
**المقاييس:** RTF (Real-Time Factor) = وقت المعالجة / مدة الصوت؛ أقل من 1 يعني أسرع من الـ real-time. للـ batch transcription المهم الـ throughput (ساعات audio لكل GPU لكل ساعة). للـ streaming المهم "عدد الـ concurrent streams اللي GPU واحد يخدمها مع الحفاظ على p95 latency أقل من X"، وده الرقم اللي لازم يطلع من benchmark مش من ورقة.

**الـ streaming models (Conformer RNN-T/CTC):** كل stream بيبعت chunk كل 160 إلى 640 ms، والـ server بيعمل dynamic batching للـ chunks من streams مختلفة في forward pass واحد. Triton بيدعم ده بالـ sequence batching (stateful models بتحتفظ بالـ encoder cache لكل stream). NVIDIA Riva مبني على الفكرة دي. الـ batch size بيتحدد بالـ tradeoff بين الـ GPU utilization والـ queueing latency.

**الـ Whisper:** faster-whisper (CTranslate2) بـ int8/fp16 مع batched pipeline على مستوى الـ VAD segments، أو whisper.cpp للـ CPU/edge، أو TensorRT-LLM وvLLM (بيدعموا Whisper) للـ throughput الأعلى مع continuous batching في الـ decoder. الـ decoder autoregressive فالـ KV cache وطول الـ output بيحكموا الـ throughput.

**التفاصيل اللي بتفرق:** VAD على CPU قبل الـ GPU عشان ما تبعتش صمت، model warmup عند الـ startup، fp16 افتراضي وint8 لو الجودة سمحت، pinned memory وتقليل الـ copies، autoscaling على عدد الـ active streams مش على GPU utilization، وhealth checks على الـ latency. للـ on-prem: MIG على A100/H100 لعزل الخدمات الصغيرة.

**سؤال متابعة:** ليه الـ GPU utilization ممكن تكون 30% والـ latency عالية في نفس الوقت؟ (batching صغير وoverhead في الـ Python/serialization، مش الـ compute).

### س94. الـ TTS serving: ازاي توفّق بين الـ throughput والـ latency؟ تكلفة الـ vocoder، الـ batching للـ AR models، والـ caching.

**الإجابة النموذجية:**
الـ TTS الحديثة مكوّنة من مرحلتين بتكلفة مختلفة: الـ AR token generation (زي LLM: بيتبنى عليه continuous batching وKV cache، وممكن يتخدم بـ vLLM مباشرة لو الموديل Llama-based زي Orpheus)، والـ mel/waveform decoding (flow matching بعدد NFE steps، ثم vocoder). الـ flow matching ممكن يتسرّع بتقليل الـ steps أو distillation أو consistency models، والـ vocoder (Vocos/HiFi-GAN) رخيص نسبيًا ويتعمل له batching بسهولة.

**الـ latency:** fast path لأول chunk (نص قصير، أولوية عالية في الـ scheduler)، وstreaming للباقي. **الـ throughput:** batching على مستوى الجمل من طلبات مختلفة، وpadding-aware bucketing حسب طول النص.

**الـ caching:** في الـ IVR والـ agents فيه نسبة كبيرة من الجمل ثابتة (الترحيب، القوائم، رسائل الخطأ). خزّنها pre-rendered بمفتاح hash(text + voice + model version) في object storage أو Redis، وده بيوفر GPU ويدي latency صفر. للجمل الديناميكية جزئيًا (رصيدك 500 ريال) ممكن تجميع segments مع crossfade، لكن الجودة أقل.

**التكلفة:** موديلات صغيرة (Kokoro، Piper) بتشتغل على CPU بجودة مقبولة لبعض الحالات، فممكن tiering: صوت premium على GPU للـ brand voice وCPU للـ fallback. قيس streams/GPU عند الـ TTFB المستهدف زي الـ STT.

**سؤال متابعة:** لو نفس الجملة اتطلبت بصوتين مختلفين، الـ cache بيتقسم ازاي؟ وازاي تعمل invalidation لما تحدّث الموديل؟

### س95. الـ quantization والـ optimization لموديلات الكلام: ايه اللي بينفع وايه اللي بيتكسر؟ والـ on-device.

**الإجابة النموذجية:**
**في الـ STT:** Whisper بيتحمل int8 (weights وactivations) في CTranslate2 بخسارة WER شبه معدومة، وfp16 دايمًا آمن، وfp8 على Hopper بيدي throughput أعلى. الـ Conformer كذلك. الـ distillation (distil-whisper) بتقلل الـ decoder layers لكنها متعملة للإنجليزي؛ للعربي لازم تعمل distillation بنفسك أو تستخدم turbo. الـ pruning أقل شيوعًا.

**في الـ TTS:** الـ acoustic model/LLM part بيتحمل fp16 وint8 غالبًا، لكن الـ vocoders والـ flow-matching decoders حساسة: الـ quantization ممكن يطلع artifacts مسموعة (buzzing، metallic sound) رغم إن الأرقام تبان كويسة، فلازم listening test بعد أي optimization مش بس WER/UTMOS.

**الـ on-device/edge:** whisper.cpp وsherpa-onnx (Zipformer، Paraformer) للـ STT، وPiper/Kokoro للـ TTS بـ ONNX، وSilero VAD. بيبقى السؤال memory وbattery وثبات الـ latency على CPU. مفيد لو الـ privacy أو الـ offline requirement، ومناسب لـ kiosks وتطبيقات الموبايل.

**القاعدة:** بعد كل optimization قيس التلاتة مع بعض: الـ speed، الـ WER/MOS delta على test set كامل، والـ tail latency. وخلي عندك golden set صغير بيتسمع يدويًا للـ TTS.

**سؤال متابعة:** ليه الـ int8 على الـ decoder بتاع Whisper ممكن يزود الـ hallucination في حالات نادرة؟ (الـ logits الصغيرة بتتغير والـ no_speech/temperature fallback بيتأثر).

### س96. الـ on-prem deployment: ايه اللي بيتغير في التصميم؟ الـ licenses والـ hardware والتحديثات.

**الإجابة النموذجية:**
**الـ licenses قبل أي حاجة:** open weights مش معناها commercial use. أمثلة (لازم تتراجع وقت القرار لأنها بتتغير): Whisper بـ MIT، NeMo Parakeet/Canary بـ CC-BY-4.0، CosyVoice بـ Apache 2.0، Orpheus وKokoro بـ Apache 2.0، Piper بـ MIT، أما F5-TTS فالكود MIT لكن الـ pretrained weights متدربة على Emilia وبالتالي CC-BY-NC (غير تجاري) وتحتاج تعيد التدريب على data مرخصة، وXTTS تحت Coqui Public Model License (غير تجاري)، وFish Speech بـ CC-BY-NC-SA. الـ datasets نفسها لها licenses كمان، والنقطة دي بتقتل مشاريع.

**الـ hardware:** sizing من الـ benchmarks بتاعتك، N+1 redundancy، MIG لتقسيم GPUs بين خدمات صغيرة، CPU كافي للـ media server والـ VAD، وNVMe للـ model loading. مفيش autoscaling لا نهائي، فالـ capacity planning بيبقى محافظ وفيه queueing policy واضحة (ايه اللي يحصل عند الذروة: rejection، degradation لموديل أصغر، أو تحويل لموظف).

**الـ air-gapped:** الموديلات والـ containers بتتنقل يدويًا، محتاج model registry داخلي، pipeline للـ validation قبل النشر (WER/MOS regression على golden set)، وrollback سريع. الـ monitoring (Prometheus/Grafana/OpenTelemetry) self-hosted، والـ LLM evaluation كمان self-hosted.

**الـ data residency والأمان:** كل الـ audio والـ transcripts جوه الـ DC، تشفير، RBAC، وسياسات retention. وفي السعودية الـ PDPL وأحيانًا متطلبات الجهة (SAMA للبنوك، NCA للحكومة) بتحدد حتى مكان الـ backups.

**سؤال متابعة:** لو الـ vendor بتاع الـ LLM قدّم موديل on-prem بـ license لعدد GPUs محدد، ازاي تصمم عشان تقلل العدد؟ (quantization، prefix caching، تقليل الـ context، موديل أصغر للمحادثة).

### س97. حسابات الـ capacity planning: عايز 1000 مكالمة متزامنة. ازاي تحسب عدد الـ GPUs للـ STT والـ TTS والـ LLM، وايه الأرقام اللي لازم تقيسها بنفسك؟

**الإجابة النموذجية:**
المدخل الأساسي هو الـ concurrency في الذروة مش عدد المكالمات في اليوم، وكل مكوّن له وحدة قياس مختلفة: (1) الـ streaming STT: عدد الـ streams المتزامنة لكل GPU عند latency مقبولة. موديل CTC/TDT بحجم 0.6B على GPU حديث بيشيل مئات الـ streams لما الـ server بيعمل batching على الـ chunks (الـ compute لكل stream صغير لأن كل chunk 100-500 ms)، بينما Whisper large في وضع pseudo-streaming بيشيل عشرات بس. (2) الـ TTS: الـ AR LLM-style model محدود بالـ decoding steps (عشرات الـ tokens في الثانية لكل stream) وبالـ memory bandwidth، فالـ GPU بيشيل عشرات الـ streams المتزامنة مع batching، والـ NAR أكتر؛ بس مش كل المكالمات بتتكلم في نفس الوقت (الـ agent بيتكلم حوالي 30-40% من وقت المكالمة، والباقي سمع وصمت)، فالـ TTS concurrency الفعلية حوالي تلت المكالمات. (3) الـ LLM: الـ turns بتوصل بشكل متقطع (turn كل 10-20 ثانية لكل مكالمة)، فـ 1000 مكالمة بتعمل حوالي 50-100 request في الثانية بـ prompt طويل وoutput قصير، والـ prefix caching بيقلل الـ prefill جدًا، وموديل 7-8B على vLLM بـ fp8 على GPU واحد أو اتنين بيقدر يخدم ده بـ TTFT مقبول، لكن لازم تقيس مع طول context واقعي.

الطريقة: تبني load test بـ profile واقعي (توزيع طول الـ turns، نسبة الكلام، طول الـ context) وتزوّد الحمل تدريجيًا لحد ما p95 لأي مرحلة يكسر الـ SLO، وده رقم "الـ streams لكل GPU" الحقيقي بتاعك. وبعدين: headroom 30-50% للذروة غير المتوقعة، N+1 للأعطال، وفصل الـ pools (STT وTTS وLLM على GPUs مختلفة غالبًا لأن سلوكهم مختلف: الـ LLM memory-bound والـ STT compute-bound قصير). والـ memory: KV cache للـ LLM هو اللي بيحدد الـ concurrency القصوى، والـ TTS AR كذلك.

المرشح القوي بيذكر إن الأرقام المنشورة من الـ vendors مقاسة في ظروف مثالية (إنجليزي، utterances قصيرة، latency عالية)، وإن العربي واللهجات وطول الـ prompts بيغيروا كل حاجة، وإن الليل فاضي فالـ utilization الفعلي بيكون واطي وده بيفتح نقاش الـ batch jobs في أوقات الفراغ.

**سؤال متابعة:** لو الـ budget بيسمح بنص عدد الـ GPUs المحسوب، ايه أول ثلاث حاجات تضحي بيها؟

### س98. الـ export للـ production: ONNX وTensorRT وTriton لموديلات الكلام. ايه المشاكل اللي بتظهر مع الـ streaming encoders والـ stateful models والـ dynamic shapes والـ custom ops؟

**الإجابة النموذجية:**
المشاكل النمطية: (1) الـ dynamic shapes: طول الصوت متغير، والـ TensorRT محتاج optimization profiles بحدود min/opt/max، ولو الـ utterance أطول من max بتفشل أو بتقطعها؛ والـ padding الزائد بيضيّع compute في الـ batching فبتعمل bucketing. (2) الـ stateful streaming: الـ encoder الـ chunked محتاج يحتفظ بـ cache (الـ attention keys/values السابقة وحالة الـ conv)، وده بيتعمل يا إما بإرجاع الـ cache كـ outputs وإعادة إرساله (stateless server، أبسط للـ scaling)، يا بـ Triton sequence batching (الـ server بيحتفظ بالـ state لكل stream ويوجّه الـ chunks لنفس الـ instance). (3) الـ custom ops: الـ CTC/transducer decoding والـ beam search والـ LM fusion مش ops عادية، فبتتنفذ في Python backend أو BLS في Triton أو في الـ client، وده ممكن يبقى الـ bottleneck. (4) الـ RNN-T decoding loop: الـ greedy decoding فيه while loop يعتمد على الـ output، والـ export له بيحتاج فصل الـ encoder والـ predictor والـ joiner كنماذج منفصلة والـ loop في الـ backend. (5) الـ numerics: fp16 في TensorRT ممكن يعمل overflow في الـ attention أو الـ LayerNorm، والحل تخصيص layers لـ fp32 أو استخدام bf16 لو مدعوم، ودايمًا تقارن outputs الـ exported بالـ PyTorch على test set كامل (مش على sample واحد) بالـ WER مش بالـ tensor difference. (6) الـ preprocessing: الـ mel extraction لازم يكون جزء من الـ graph أو متطابق بالظبط في الـ client (نفس المكتبة والـ parameters)، وأغلب مشاكل "الـ WER زاد بعد الـ export" أصلها هنا.

الـ TTS أصعب: الـ AR LLM-style model بيتعامل كـ LLM (vLLM أو TensorRT-LLM مع custom tokenizer وinference للـ codec decoder منفصل)، والـ flow matching decoder له عدد steps متغير، والـ vocoder سهل. ولكل ده لازم golden test: مجموعة inputs ومخرجاتها من الـ reference implementation، وأي export لازم يعدّي عليها قبل الـ deployment.

**سؤال متابعة:** ايه الفرق بين تحقق الـ export بـ "max abs difference أقل من 1e-3" وبين تحققه بالـ WER؟ وامتى الأول بيخدعك؟

### س99. صمم الـ streaming API لخدمة STT داخلية: البروتوكول (gRPC bidirectional مقابل WebSocket)، شكل الرسائل (partial/final، timestamps، confidence)، الـ backpressure، وإدارة الـ sessions.

**الإجابة النموذجية:**
الاختيار: gRPC bidirectional streaming للـ service-to-service (typed، فعال، وفيه flow control مدمج)، وWebSocket للـ clients (المتصفح والـ mobile وTwilio-style integrations). بعض الفرق بتعمل الاتنين على نفس الـ core.

شكل الرسائل من الـ client: رسالة config أولًا (sample rate وencoding وlanguage وenable partials وhotwords وsession id) ثم audio chunks (20-100 ms، raw PCM 16-bit أو Opus) ثم end-of-stream. من الـ server: partial (نص مؤقت للـ segment الحالي، بيتبدّل)، final (نص ثابت للـ segment مع بداية ونهاية بالثواني، وwords مع timestamps وconfidence، والنص الخام والمطبّع)، وأحداث: speech started، speech ended (endpoint)، وerrors، وtimeouts. لازم كل final له id متزايد عشان الـ client يقدر يعمل reconciliation، ولازم توثّق إن الـ partials مش مضمونة تكون prefix للـ final.

الـ backpressure: لو الـ server بطيء، الـ client لازم يعرف: في gRPC الـ flow control بيبطّئ الإرسال تلقائيًا، وفي WebSocket بتراقب الـ buffered amount وبترمي أو بتضغط لو زاد، والـ server بيرفض sessions جديدة (503 مع retry-after) لما الـ capacity توصل حدها بدل ما كل الـ sessions تتدهور. كمان حد أقصى لمدة الـ session وحجم الـ buffer، وقياس الـ "audio lag" (الفرق بين وقت الصوت المستلم ووقت المعالج) كـ metric.

إدارة الـ sessions: كل session مرتبطة بـ instance معين (sticky) لأن الـ state موجود هناك، فالـ load balancer لازم يكون على مستوى الـ connection مش الـ request، والـ reconnect بيحتاج إما إعادة إرسال آخر ثواني من الصوت أو قبول فقدان جزء. والأمان: auth token في أول رسالة أو في الـ header، وquota لكل tenant، وlogging بدون تخزين الصوت افتراضيًا.

**سؤال متابعة:** الـ client بيبعت chunks بحجم 20 ms والـ server بيعالج كل 200 ms. فين الـ buffering المفروض يحصل وليه؟

### س100. تشغيل speech workloads على Kubernetes: الـ GPU scheduling، الـ autoscaling على active streams، الـ graceful draining للمكالمات الطويلة، وتحميل الموديلات. ايه اللي بيفرق عن microservice عادي؟

**الإجابة النموذجية:**
الفروق الجوهرية: (1) الـ pod ما ينفعش يتقفل فجأة: مكالمة ممكن تستمر 20 دقيقة والـ state جوه الـ pod، فالـ terminationGracePeriod لازم يكون طويل، والـ pod يدخل وضع draining (يرفض sessions جديدة ويكمّل الحالية) عند SIGTERM، والـ deployment يعمل rolling update ببطء. (2) الـ autoscaling على الـ CPU/memory ما ينفعش؛ بتعمل scaling على custom metrics (عدد الـ active streams، الـ GPU utilization، طول الـ queue) عبر Prometheus adapter أو KEDA، مع scale-up سريع وscale-down بطيء وحد أدنى كبير لأن تحميل موديل على GPU بياخد دقايق. (3) الـ readiness: الـ pod ما يستقبلش traffic إلا بعد تحميل الموديل والـ warmup (أول inference بطيء بسبب الـ CUDA kernels والـ TensorRT)، فالـ readiness probe بتعمل inference فعلي. (4) الـ GPU: الـ device plugin وGPU operator، وnode pools منفصلة حسب نوع الـ GPU، وقرار MIG (تقسيم GPU كبير لأجزاء معزولة، مناسب للـ STT الصغير) مقابل time-slicing (مشاركة بدون عزل، خطر noisy neighbor) مقابل pod واحد يشغّل server بيعمل batching داخلي (الأفضل غالبًا للـ throughput). (5) الموديلات: الـ weights مش جوه الـ image (بتكبّرها لجيجابايتات وبتبطّئ الـ pull)، لكن على PVC مشترك أو object store داخلي بيتحمّل بـ init container مع hash للتحقق. (6) الـ stickiness: الـ streams لازم تفضل على نفس الـ pod، فالـ ingress بيدعم long-lived connections والـ timeouts طويلة. (7) الـ observability على مستوى الـ GPU (DCGM exporter) والـ stream.

وفي الـ on-prem الـ cluster صغير وثابت غالبًا، فالـ autoscaling بيبقى بين الخدمات (تدي الـ LLM GPUs أكتر في الذروة والـ batch jobs بالليل) أكتر من ما هو إضافة nodes.

**سؤال متابعة:** عملت rolling update للـ STT service والمكالمات القديمة اتقطعت رغم الـ grace period. ايه الأسباب المحتملة؟ (الـ ingress قطع الـ connection، الـ preStop hook ناقص، الـ service endpoints اتشالت فالـ reconnect فشل).

### س101. تحديث الموديلات في الـ production الصوتي: canary وA/B وshadow، معايير الـ rollback، وليه الـ TTS voice versioning له حساسية خاصة؟

**الإجابة النموذجية:**
كل موديل (STT، TTS، LLM، VAD، turn detector) له version مستقل وconfig مستقل (normalizer، lexicon، thresholds)، والـ deployment بيسجّل الـ combination الكاملة لكل مكالمة عشان تقدر تعزو أي تغيير في الـ metrics. الإطلاق: (1) offline gates: الموديل الجديد لازم يعدّي الـ test sets الأساسية والـ regression set (مكالمات حقيقية معروفة المشاكل) بدون تدهور في أي slice مهم، مش بس تحسن في المتوسط. (2) shadow: بيشوف traffic حقيقي بدون تأثير وبتقارن مخرجاته بالحالي (للـ STT: الفرق بين الـ transcripts، وعينة تتراجع بشريًا). (3) canary: 1-5% من المكالمات بمراقبة metrics على مدى يومين على الأقل عشان تشوف الذروة، مع rollback تلقائي لو أي metric حرج (error rate، latency p95، handoff rate) كسر threshold. (4) A/B لما الفرق subtle (containment، رضا) مع حجم عينة كافي.

الـ rollback لازم يكون بضغطة زر وبيرجّع الـ config كمان مش الـ weights بس، وتحتفظ بالنسختين محملين وقت الـ canary.

الـ TTS voice: العميل بيعرف "صوت الشركة"، وأي تغيير في الـ timbre أو الإيقاع بيتلاحظ فورًا حتى لو الـ MOS اتحسن. فالـ voice version جزء من الـ brand: تحديث الموديل لازم يحافظ على الهوية (speaker similarity مع النسخة السابقة كمقياس)، والـ pre-rendered prompts لازم تتعاد بالكامل بالنسخة الجديدة عشان ما يبقاش فيه صوتين في نفس المكالمة، والـ cache لازم يتفرّغ بالإصدار، ويفضّل تحديث الـ voice مع إعلان داخلي وربما اختبار مع عملاء.

وفي الـ STT: تغيير الـ normalizer أو الـ ITN تحديث "موديل" برضه لأنه بيغيّر المخرجات اللي الـ LLM والـ tools بتعتمد عليها، ولازم يعدّي نفس الـ gates.

**سؤال متابعة:** الـ canary بيبان أحسن في كل الـ metrics بس الشكاوى زادت. ايه اللي ممكن يكون حصل؟ (الـ canary بيروح لشريحة مختلفة من المستخدمين، أو الـ metric المهم مش مقاس).

### س102. الـ on-device speech (موبايل، سيارة، أجهزة داخل الفروع): ايه الأدوات والموديلات، وايه القيود (حجم، latency، بطارية، NPU)، وايه التصميم الهجين مع الـ server؟

**الإجابة النموذجية:**
الأدوات: whisper.cpp (GGML مع quantization وتسريع Metal/CUDA، Whisper بأحجام tiny لـ large)، sherpa-onnx (من فريق k2: Zipformer streaming وParaformer وSenseVoice وWhisper للـ STT، وPiper وMatcha وKokoro للـ TTS، وSilero VAD، على Android وiOS وLinux embedded)، Vosk (Kaldi) القديم لكن خفيف، Moonshine للـ streaming الخفيف بالإنجليزي، WhisperKit على أجهزة Apple بيستغل الـ Neural Engine. القيود: (1) الحجم: موديل 100-300 MB مقبول في تطبيق، والجيجابايت لأ. (2) الـ latency والطاقة: الـ CPU بيخلي الموديل يشتغل بس بيسخّن ويستهلك بطارية، والـ NPU (Apple ANE، Qualcomm Hexagon) أوفر بكتير لكن بيحتاج export لصيغ خاصة (Core ML، QNN) وبيدعم ops محدودة. (3) التنوع: مئات أنواع الأجهزة بأداء مختلف، فبتحتاج أكتر من حجم موديل واختيار runtime. (4) التحديث: الموديل بيتنزل مع التطبيق أو كـ download منفصل، وصعب تعمل rollback سريع.

للعربي: الموديلات الصغيرة الجاهزة (Whisper tiny/base) ضعيفة جدًا في العربي اللهجي، فالمسار العملي تدريب/تقطير موديل صغير خاص (Zipformer أو FastConformer بحجم 30-100M) على الـ data بتاعتك ثم export لـ sherpa-onnx بـ int8. والـ TTS الصغير محتاج صوت عربي متدرب بنفسك (Piper/Matcha).

التصميم الهجين: على الجهاز الـ wake word والـ VAD والأوامر القصيرة والـ TTS للإشعارات (خصوصية وسرعة وعمل بدون شبكة)، وعلى الـ server المحادثات الطويلة والـ LLM. ولازم مسار fallback واضح لما الشبكة تقطع، وقياس الجودة على الجهاز الحقيقي مش على الـ simulator.

**سؤال متابعة:** الموديل بيشتغل ممتاز على iPhone حديث وبطيء جدًا على أجهزة Android متوسطة. ايه استراتيجية المنتج؟

### س103. الـ offline transcription بحجم آلاف الساعات يوميًا (أرشيف مكالمات، compliance): ازاي تصمم الـ pipeline للـ throughput والتكلفة، وايه الـ metrics؟

**الإجابة النموذجية:**
هنا الـ latency مش مهمة، الـ throughput والتكلفة لكل ساعة صوت هما اللي بيفرقوا. التصميم: (1) pipeline على مراحل بـ queue بينهم (Kafka أو SQS-style أو حتى DB jobs): استلام الملف، تحويل الصيغة والتحقق (ffmpeg، فصل القنوات، sample rate)، VAD وتقطيع لـ segments (على CPU، رخيص)، STT على GPU بـ batches كبيرة من الـ segments مرتبة حسب الطول (عشان الـ padding يقل)، ثم post-processing (ITN، punctuation، diarization لو محتاج، redaction) على CPU، ثم التخزين والفهرسة. (2) الـ batching: الفرق بين معالجة ملف ملف وبين تجميع segments من ملفات كتير في batch واحد ممكن يوصل لـ 10x في الـ throughput؛ Whisper large بـ batched decoding (WhisperX style) أو موديل CTC/TDT اللي أصلًا أسرع بكتير. (3) الـ metrics: ساعات صوت معالجة لكل ساعة GPU (وبالتالي التكلفة لكل ساعة صوت)، والـ GPU utilization (لو أقل من 70% يبقى الـ CPU stages أو الـ IO هي الـ bottleneck)، والـ backlog age (أقدم ملف مستني)، ونسبة الفشل. (4) الـ scheduling: تشغيل الـ backlog بالليل على الـ GPUs بتاعة الـ real-time لما تكون فاضية. (5) الـ idempotency والـ retries: كل job له id وحالة، وإعادة المعالجة ما تعملش duplicates. (6) الجودة: عينة يومية للمراجعة، وقياس WER على subset معنون.

للامتثال في مركز اتصال بنكي: القنوات المنفصلة بتلغي الـ diarization، والـ redaction للأرقام الحساسة بتتعمل قبل التخزين الدائم، والفهرسة بتسمح بالبحث عن كلمات (شكوى، إلغاء) وبتغذّي الـ QA والـ analytics.

**سؤال متابعة:** الـ throughput نزل للنص فجأة بعد تحديث. ايه أول حاجة تشوفها؟ (الـ batch size الفعلي، توزيع أطوال الـ segments، الـ VAD بيقطّع أصغر، أو الـ GPU utilization).

### س104. مشاركة الـ GPUs بين STT وTTS وLLM في cluster صغير: pools منفصلة ولا مشتركة؟ MIG مقابل MPS مقابل time-slicing، ومشكلة الـ noisy neighbor.

**الإجابة النموذجية:**
كل workload له طبيعة: الـ LLM decoding memory-bandwidth-bound وبيحتاج KV cache كبير ومستمر؛ الـ STT streaming بيعمل bursts صغيرة كتير compute-bound؛ الـ TTS AR شبه الـ LLM لكن أصغر، والـ vocoder/flow decoder compute-bound قصير. لما تحطهم على نفس الـ GPU بدون عزل، الـ LLM بيستهلك الـ memory والـ STT chunks بتستنى في الطابور ورا kernels طويلة، فالـ latency p95 للـ STT بتتضاعف في الذروة (noisy neighbor). عشان كده الافتراضي pools منفصلة حتى لو صغيرة، والمشاركة بس لما الحجم صغير جدًا أو للـ batch jobs.

خيارات المشاركة: (1) MIG (على A100/H100): تقسيم الـ GPU لأجزاء بعزل حقيقي في الـ memory والـ compute؛ ممتاز للـ STT والـ TTS الصغار (instance 10-20 GB لكل خدمة)، بس بيقلل الـ peak throughput وما ينفعش مع الـ LLM الكبير. (2) MPS: عدة processes بتشارك الـ GPU بـ kernels متزامنة، أحسن utilization من الـ time-slicing، بس مفيش عزل للـ memory والـ failure في process ممكن يأثر على الباقي. (3) time-slicing الافتراضي في Kubernetes: أبسط وأسوأ عزل. (4) server واحد multi-model (Triton بيشغّل عدة models على نفس الـ GPU مع priorities وrate limits): بيدي تحكم أحسن من الـ OS-level sharing.

القاعدة: ابدأ بـ pools منفصلة بأصغر حجم، وقيس الـ utilization الفعلي، ولو الـ STT GPU فاضي 80% من الوقت حط عليه الـ batch jobs أو TTS خفيف بـ priority أقل، مش الـ LLM. والـ observability لكل model على الـ GPU (queue time وcompute time منفصلين) هي اللي بتكشف الـ contention.

**سؤال متابعة:** عندك 4 GPUs بس لـ pilot. ازاي توزّع STT وTTS وLLM؟ وليه ممكن تحط الـ VAD والـ turn detector على CPU؟

---

## 7. الـ data

### س105. عايز تبني dataset لصوت TTS جديد بلهجة سعودية. التسجيل والـ script والـ annotation والـ QA. ايه اللي بيخلي الـ dataset كويس؟

**الإجابة النموذجية:**
**المتكلم:** صوت واحد محترف (voice actor أو مذيع)، ثابت الأداء، من نفس اللهجة المستهدفة، مع عقد واضح بيغطي حقوق استخدام الصوت والـ cloning والمدة والاستخدامات (مهم قانونيًا وأخلاقيًا).

**التسجيل:** استوديو معزول، نفس المايك والمسافة في كل الجلسات، 48 kHz/24-bit، جلسات قصيرة (ساعتين كحد أقصى) عشان الصوت ما يتعبش ويتغير، وrecording engineer بيراجع الأخطاء وقتها. الحجم: 5 إلى 10 ساعات كافية لـ fine-tuning موديل pretrained قوي، وأقل من ساعة لـ zero-shot adaptation، و20+ ساعة لو هتدرب من الصفر. الجودة أهم من الكمية.

**الـ script:** phonetically balanced (يغطي كل phonemes العربي في سياقات مختلفة: الحروف الحلقية، الشدة، المد)، متنوع في الطول (جمل قصيرة وطويلة)، أنواع الجمل (سؤال، أمر، تعجب)، ومحتوى من الـ domain (بنكي، خدمة عملاء) مع أرقام وتواريخ وأسماء وcode-switching إنجليزي بنسبة واقعية. أسلوب واحد ثابت (neutral/friendly) إلا لو هتبني expressive dataset فتحتاج labels.

**الـ annotation:** الـ transcript لازم يطابق اللي اتنطق فعلًا (مش الـ script الأصلي، لأن المتكلم بيغير كلمات)، مشكّل بالكامل ومراجع من لغوي بنفس اللهجة، وبقواعد كتابة موحدة للهجة (وثيقة conventions). ده أغلى جزء وأكثره تأثيرًا.

**الـ QA والمعالجة:** قياس SNR (فوق 35 dB) وDNSMOS لكل ملف، إزالة الـ clipping والنفس العالي والضوضاء، forced alignment لتقطيع الجلسات لـ utterances من 1 إلى 15 ثانية والتأكد من مطابقة النص، تنظيم الصمت في البداية والنهاية (100 إلى 200 ms)، وتوحيد الـ loudness. وكمان holdout set للتقييم.

**سؤال متابعة:** لو المتكلم أداؤه اتغير بين الجلسة الأولى والخامسة، ازاي تكتشف ده وتتعامل معاه؟

### س106. ازاي تجمع آلاف الساعات لـ STT بتكلفة معقولة؟ found data، pseudo-labeling، synthetic data، الـ licensing.

**الإجابة النموذجية:**
**الـ found data مع نص:** محتوى فيه transcripts أو subtitles (أخبار، برامج، محاضرات، مجالس تشريعية، بودكاست فيه show notes). الـ transcripts بتكون تقريبية، فبتعمل long-audio alignment (CTC segmentation) وتحتفظ بالـ segments اللي الـ alignment score بتاعها عالي، وتصلّح الباقي بموديل. الـ licensing هنا الفخ الأكبر: شروط استخدام المنصات (YouTube مثلًا بيمنع الـ scraping)، وحقوق المحتوى، فمحتاج legal review أو اتفاقيات مع الناشرين.

**الـ pseudo-labeling:** موديل قوي بيكتب transcripts لـ unlabeled audio، وبتفلتر بالـ confidence وبالاتفاق بين موديلين مختلفين (Whisper مقابل Conformer)، وتدرب موديل student على الـ mix (noisy student training). ده أكبر رافعة عمليًا، ولازم تعمل iterations. الخطر: تثبيت أخطاء الـ teacher (خصوصًا الهمزات والأرقام)، فبتحتفظ بنسبة human-labeled كـ anchor.

**الـ human-labeled data:** crowdsourcing لقراءة نصوص (read speech زي Common Voice) رخيص لكن مش conversational، وتسجيل محادثات موجهة (prompted dialogues بين اتنين) أغلى لكن أقرب للواقع. والـ annotation لمكالمات حقيقية (بموافقة) هو الأغلى والأهم للـ call center.

**الـ synthetic data بالـ TTS:** مفيد لتغطية كلمات نادرة وأسماء منتجات وأرقام، وللـ domain adaptation على مستوى الـ LM، لكن التنوع الصوتي محدود، وكتير منه بيضر لأن الموديل بيتعلم "صوت TTS". نسبة صغيرة ومع augmentation.

**الإدارة:** guidelines مكتوبة، قياس inter-annotator agreement، dashboards للـ annotators، وتوازن اللهجات والجنس والعمر والقنوات في الـ metadata عشان تقدر تقيس per-slice.

**سؤال متابعة:** ازاي تكتشف إن الـ pseudo-labels بتدهور جودة الموديل على slice معين رغم إن الـ WER الكلي بيتحسن؟

### س107. ايه القرارات الأساسية في transcription guidelines للهجات السعودية؟

**الإجابة النموذجية:**
الـ guidelines بتحدد جودة الـ data أكتر من أي حاجة، ولازم تتكتب قبل التعاقد مع annotators:

1. **الـ orthography:** كتابة اللهجة زي ما اتنطقت بحروف عربية بأقرب شكل للفصحى (مثلًا "وش" و"ليش" و"ابغى") مع قائمة كلمات موحدة للكلمات المتكررة، وقرار واضح في الهمزات والتاء المربوطة والألف المقصورة.
2. **الفصحى داخل اللهجة:** تتكتب زي ما اتنطقت، من غير "تصحيح" لفصحى.
3. **الـ code-switching:** الكلمات الإنجليزية بحروف لاتينية (لو الـ STT هيطلع لاتيني) ولا بحروف عربية؟ قرار واحد للـ dataset كله، ومرتبط بالـ product (ايه اللي المستخدم عايز يشوفه). الأسماء التجارية غالبًا بشكلها الرسمي.
4. **الأرقام:** بالحروف (spoken form) في الـ transcript الخام، والـ ITN يطبّق بعدين، عشان الموديل يتعلم اللي اتنطق فعلًا. أو العكس مع ثبات.
5. **الظواهر الصوتية:** الـ fillers (اممم، يعني) بتتكتب ولا لأ (للـ turn detection مفيدة)، التكرار والتلعثم، الكلام غير المسموع بعلامة [غير مفهوم]، الضحك والضوضاء بعلامات ثابتة، الكلام المتداخل.
6. **الترقيم والتشكيل:** غالبًا من غير تشكيل، وترقيم بسيط أو بدون حسب الموديل.
7. **الـ PII:** إخفاء أو وسم البيانات الحساسة في الـ transcript والـ audio.
8. **الجودة:** double annotation على 5 إلى 10% لقياس الاتفاق، مراجعة لغوي، وأمثلة كتير في الوثيقة لكل حالة.

**سؤال متابعة:** لو عندك annotators من لهجات مختلفة بيكتبوا نفس اللهجة، ايه المشكلة وازاي تحلها؟

### س108. إدارة عملية الـ annotation لـ 1000 ساعة لهجات سعودية: الأدوات، تدريب الـ annotators، الـ throughput والتكلفة، الـ QA والـ inter-annotator agreement، والـ feedback loop مع الـ guidelines.

**الإجابة النموذجية:**
الـ annotation مشروع تشغيلي قبل ما يكون تقني. المكونات: (1) الأداة: Label Studio أو أداة داخلية بواجهة بسيطة: واجهة تعرض الـ segment (5-15 ثانية) مع الـ waveform وتسمح بالتشغيل بلوحة المفاتيح، وحقل النص، وtags للحالات الخاصة (غير مفهوم، ضوضاء، متكلم تاني، code-switch)، وpre-filling بـ transcript من الموديل الحالي (بيزوّد السرعة 2-3x بس بيعمل bias، فلازم تقيس الفرق). (2) الـ annotators: ناطقين باللهجة نفسها (مش أي عربي)، تدريب على الـ guidelines بأمثلة صوتية حقيقية وامتحان قبل البدء، وحد أدنى من الاتساق قبل ما شغلهم يدخل الـ dataset. (3) الـ throughput: النسخ الدقيق للمكالمات الطبيعية بياخد 6-10 مرات مدة الصوت (ساعة صوت = 6-10 ساعات عمل) وأقل مع الـ pre-filling والصوت النظيف؛ وده اللي بيحدد التكلفة والجدول. (4) الـ QA: عينة 5-10% من كل annotator بتتراجع من reviewer أقدم، وdouble annotation لنسبة عشان تقيس الـ agreement (WER بين الاتنين بعد normalization؛ لو فوق 10-15% في اللهجات فالـ guidelines مش واضحة)، وgold set صغير معروف الإجابة بيتحقن دوري لقياس كل annotator، وdashboard لسرعة ودقة كل واحد. (5) الـ feedback loop: أسبوعيًا تجمع الحالات المختلف عليها وتحسم قرار في الـ guidelines وتبلّغ الكل، والـ guidelines document بيتطور بأمثلة.

الحاجات اللي بتفرق في الجودة أكتر من الأداة: الـ segments قصيرة ونظيفة الحدود، الصوت بيتوزع بحيث كل annotator ياخد لهجته، والدفع بالساعة مش بالـ segment (الدفع بالـ segment بيدفع للسرعة على حساب الدقة)، وتوثيق كل قرار.

**سؤال متابعة:** annotator سريع جدًا ودقته على الـ gold set ممتازة، بس الـ reviewer بيلاقي أخطاء كتير في شغله. ايه التفسير المحتمل؟ (الـ gold set مكشوف أو سهل، أو بيعتمد على الـ pre-fill من غير سماع).

### س109. صمم test set مرجعي لـ Arabic STT على مستوى الشركة: التغطية، الحجم، طريقة النسخ، الإدارة، ومنع التسريب للتدريب.

**الإجابة النموذجية:**
الـ test set هو "الميزان" اللي كل قرار هيتقاس بيه، فبيتبني بعناية أكتر من الـ training data. التغطية: matrix من الأبعاد: اللهجة (نجدي، حجازي، جنوبي، شمالي، شرقي، ومصري وشامي لو السوق بيشملهم، وMSA)، القناة (تليفون 8k، موبايل، استوديو، اجتماع)، نوع الكلام (مقروء، عفوي، مكالمة حقيقية)، الجنس والعمر، الضوضاء، والـ domain (بنكي، اتصالات، عام)، مع code-switching وأرقام وأسماء بنسب واقعية. الحجم: عشان تكتشف فرق 1% مطلق في الـ WER بثقة محتاج آلاف الجمل في كل slice مهم، فالمجموع بيوصل لعشرات الساعات، وده استثمار مبرر.

النسخ: double annotation كامل (كل ملف من اتنين مستقلين) مع adjudication من ثالث للاختلافات، بـ guidelines أكثر تفصيلًا من التدريب، والنص بيتحفظ بصورتين: verbatim خام وnormalized. الإدارة: الـ test set مقفول (access محدود، ومش موجود في أي bucket بيتقرا منه الـ training)، وليه dev set مقابل بنفس التصميم للتجارب اليومية، وversioning للـ set نفسه (لو صلّحت transcript لازم كل الأرقام القديمة تتعاد). منع التسريب: الـ speakers في الـ test ما يظهروش في الـ training (speaker-disjoint)، والنصوص كمان (لو الـ test مقروء من نصوص، النصوص دي مش في الـ training)، وفحص آلي عند كل تدريب بيقارن hashes الصوت وn-grams النص بين الـ training والـ test.

كمان: subsets صغيرة "سريعة" للتجارب المتكررة، وsubset للـ hallucination (صمت وموسيقى)، وسياسة تجديد سنوية لأن الـ production بيتغير، مع الاحتفاظ بالقديم للمقارنة التاريخية. والأرقام بتتنشر داخليًا بـ report ثابت الشكل عشان الكل يقرا نفس الحاجة.

**سؤال متابعة:** الـ test set بقى معروف للفريق وبيتحسن كل إصدار بنسب أكبر من الـ production. ايه اللي حصل؟ (overfitting على الـ test بالقرارات المتكررة، والحل dev/test منفصلين وtest جديد دوريًا).

### س110. الجانب القانوني والترخيصي للـ speech data: Common Voice، الـ academic corpora، الـ found data من YouTube والبودكاست، الموافقات، PDPL، وتوريث الترخيص في الـ synthetic data.

**الإجابة النموذجية:**
كل مصدر له وضع مختلف: (1) Common Voice ترخيصه CC0 وينفع تجاريًا، بس جودته وتنوعه محدودين والعربي فيه غالبًا MSA مقروء. (2) الـ academic corpora (MGB-2/3/5، QASR، SADA، Casablanca، FLEURS): كل واحد له license، وكتير منها للبحث فقط أو بيمنع الاستخدام التجاري أو بيقيّد إعادة التوزيع؛ لازم تقرأ الـ license لكل واحد وتوثّق القرار، ولو منتج تجاري تستبعد اللي مش مسموح حتى لو أحسن data. (3) الـ found data (YouTube، podcasts، بث): تحميل الفيديوهات بيخالف شروط الخدمة غالبًا، والمحتوى محمي بحقوق نشر، والمتكلمين ما وافقوش. الشركات الكبيرة استخدمته تحت نقاش "fair use" مش محسوم، وفي السوق السعودي/الإماراتي مع منتج للبنوك والحكومة ده خطر سمعة وقانوني كبير، فالقرار العملي: الابتعاد عنه للـ training التجاري إلا مع ترخيص صريح من الناشر. (4) الـ data المجمّع بنفسك: موافقة مكتوبة من المتكلم توضح الاستخدام (training، توزيع، cloning)، وللمكالمات موافقة العميل والموظف والإعلان في بداية المكالمة، مع حق الحذف. (5) PDPL: الصوت بيانات شخصية (بل حساسة لو بتحدد الهوية بيومتريًا)، فالمعالجة محتاجة أساس قانوني وغرض محدد وتقليل بيانات وحماية وسجل معالجة، والتخزين والمعالجة داخل المملكة أو بشروط النقل. (6) الـ synthetic data: لو TTS متدرب على data بترخيص non-commercial (زي F5-TTS بـ Emilia) فالمخرجات على الأقل مشكوك فيها تجاريًا، والقاعدة الآمنة إن ترخيص الناتج ما يكونش أوسع من ترخيص أضعف حلقة.

المرشح القوي بيقول إنه بيحتفظ بـ data lineage لكل ملف (المصدر، الترخيص، الموافقة، تاريخ الجمع) وبيقدر يعيد بناء أي موديل من غير مصدر معين لو اتسحبت موافقة، وإنه بيشرك القانونيين بدري.

**سؤال متابعة:** عميل طلب حذف بياناته من الـ training data. ايه اللي تقدر تعمله فعليًا وايه اللي لأ؟ (حذف من الـ datasets وإعادة التدريب في الدورة التالية، والموديل الحالي ما ينفعش "ينسى" بشكل مضمون، ولازم ده يكون موضح في الموافقة).

### س111. كتالوج الـ augmentation لموديلات الكلام بالـ parameters المعتادة: SpecAugment، speed perturbation، noise وRIR، codec وband-limiting، والـ mixing. وايه اللي بيضر لو زوّدته؟

**الإجابة النموذجية:**
للـ STT: (1) speed perturbation بمعاملات 0.9 و1.0 و1.1 (بتغيّر السرعة والـ pitch مع بعض) بتضاعف الـ data 3 مرات وبتدي تحسن ثابت. (2) SpecAugment على الـ mel: time masking (كذا mask بطول حتى 5-10% من الـ utterance) وfrequency masking (2-3 masks بعرض حتى 15-27 bin من 80)، وtime warping نادرًا مفيد؛ الـ policies القوية بتحتاج تدريب أطول. (3) الـ additive noise من MUSAN وDNS وضوضاء مجمّعة من بيئتك بـ SNR عشوائي بين 0 و30 dB. (4) الـ RIR convolution بنسبة 20-40% من الـ batches. (5) الـ codec وband-limiting simulation (G.711، Opus عند bitrates واطية، low-pass عند 3.4 kHz). (6) الـ gain/AGC عشوائي وclipping خفيف. (7) الـ concatenation: ضم utterances لتدريب الـ long-form. (8) الـ SpecAugment لازم يكون بعد الـ normalization مش قبلها.

للـ TTS: الـ augmentation أخطر لأن الموديل بيتعلم الـ artifacts كأنها صوت: ما تعملش noise ولا speed على الـ target audio، بس ممكن على الـ speaker prompt في الـ zero-shot models عشان الموديل يبقى robust للـ prompts الرديئة (مع الحفاظ على target نظيف).

اللي بيضر: (1) الـ over-augmentation: لو كل الـ batches فيها noise وRIR وcodec مع بعض، الموديل بيبقى ممتاز على الـ noisy وبيسوء على الـ clean، فبتحتفظ بنسبة 30-50% بدون augmentation. (2) الـ SpecAugment قوي مع data قليلة بيبطّئ التقارب جدًا. (3) الـ speed perturbation بتزيح الـ pitch فبتعمل أصوات غير واقعية عند 0.8 أو 1.2. (4) الـ noise dataset لو فيها كلام (MUSAN فيه speech) بتعلّم الموديل يتجاهل كلام حقيقي. (5) الـ RIR القوي مع الـ STT الـ streaming بيضر لأن الـ reverberation بتمتد لخارج الـ chunk.

القاعدة: كل augmentation له ablation على dev set من الـ production، والـ dev set لازم يكون فيه clean وnoisy عشان تشوف الاتجاهين.

**سؤال متابعة:** ليه الـ SpecAugment بيشتغل كويس جدًا مع الـ AED والـ CTC ومفعوله أقل على الـ SSL pretraining؟ (الـ SSL أصلًا بيعمل masking كجزء من الـ objective).

### س112. الـ synthetic speech data للتدريب: امتى تولّد data بالـ TTS وبالـ voice conversion، ايه النسب الآمنة، وازاي تتجنب إن الموديل يتعلم artifacts أو يعمل "model collapse"؟

**الإجابة النموذجية:**
الحالات المفيدة: (1) كلمات ومصطلحات نادرة للـ STT (أسماء منتجات، أرقام بصيغ مختلفة، code-switching) اللي مش موجودة في الـ data الحقيقي بكميات كافية. (2) تغطية لهجة أو سيناريو جديد قبل ما يتجمع data حقيقي. (3) توليد حوارات للـ speech LLMs والـ turn detection. (4) زيادة تنوع المتكلمين بالـ voice conversion. الحالات اللي ما تنفعش: تدريب TTS على مخرجات TTS (بيورّث الـ artifacts ويقلل التنوع)، أو تعويض غياب data حقيقي للهجة تمامًا (الـ TTS نفسه ما يعرفش اللهجة كويس).

النسب: للـ STT، التجارب المنشورة والخبرة العملية بتقول إن الـ synthetic مفيد لحد نسبة 20-30% من الـ batch لما الـ TTS عالي الجودة ومتنوع الأصوات، وبعدها بيبدأ يضر الأداء على الحقيقي. وبتخلي الـ synthetic بيمر على نفس الـ augmentation chain (noise وcodec وRIR) عشان يقرب من توزيع الحقيقي، وبتستخدم أكتر من TTS system عشان الموديل ما يتعلمش بصمة نظام واحد.

الحماية من الـ artifacts والـ collapse: (1) ablation دايمًا: الموديل مع وبدون synthetic يتقيّم على data حقيقي بس. (2) classifier بسيط بيميز synthetic من real؛ لو دقته عالية جدًا فالـ artifacts واضحة والموديل هيتعلمها. (3) الـ ASR filtering للـ synthetic نفسه (STT على الناتج، لو WER عالي ارميه). (4) وسم كل ملف synthetic في الـ manifest وتتبع نسبته في كل تدريب. (5) عدم إدخال synthetic في الـ test/dev أبدًا. (6) للـ pseudo-labeled data (صوت حقيقي بنص من موديل) ده مختلف عن synthetic (صوت مولّد)، وأغلب المكاسب الحقيقية جاية من الأول لأن الصوت حقيقي.

**سؤال متابعة:** عايز data لأرقام الهواتف بلهجات مختلفة. ازاي تولّدها بحيث تغطي طرق النطق المختلفة (تجميع الأرقام) مش بس الأرقام نفسها؟

### س113. إدارة الـ speech datasets على نطاق كبير: الـ manifests، الـ sharding، صيغ التخزين (WAV مقابل FLAC مقابل Opus)، الـ dataloaders بالـ bucketing، والـ versioning.

**الإجابة النموذجية:**
الـ manifest هو مصدر الحقيقة: ملف JSONL (NeMo style) أو Lhotse cuts فيه لكل segment: مسار الصوت والبداية والمدة والنص الخام والمطبّع والمتكلم واللهجة والمصدر والترخيص وأي metadata (SNR، channel). الـ Lhotse بالذات بيدي أدوات لـ cuts وfeatures وdynamic bucketing جاهزة.

التخزين: ملايين الملفات الصغيرة بتقتل أي file system وobject store (latency لكل ملف)، فبتعمل sharding: tar shards بحجم 1 GB تقريبًا (WebDataset style) أو Lhotse shar، وبتقرا sequentially بـ shuffle على مستوى الـ shards ثم buffer داخلي. الصيغة: WAV سهل لكن كبير (16 kHz 16-bit = 115 MB للساعة)؛ FLAC بيضغط 40-50% بدون خسارة وبيتفك بسرعة، وهو الخيار الافتراضي للـ training data؛ Opus بيضغط 10x لكنه lossy وبيغيّر الصوت (وبيحتاج decode أبطأ)، ينفع للـ found data الـ noisy أصلًا أو للأرشيف، ومش للـ TTS data. والـ features المحسوبة مسبقًا (mel) بتوفر CPU في التدريب بس بتمنع الـ waveform augmentation، فأغلب الفرق بتحسب الـ mel on-the-fly على GPU.

الـ dataloader: الـ utterances مختلفة الطول، فالـ batching بعدد ثابت بيضيّع compute في الـ padding؛ الـ dynamic batching بحد أقصى لإجمالي الثواني (max_duration) مع bucketing حسب الطول بيوصل لـ utilization أعلى بكتير ويقلل الـ OOM. مع عدد workers كافي (الـ decoding والـ augmentation على CPU غالبًا هي الـ bottleneck الفعلي في تدريب الكلام، مش الـ GPU).

الـ versioning: كل dataset له version (hash للـ manifest) وكل تدريب بيسجّل الـ versions المستخدمة، والتغييرات (إضافة مصدر، تصليح transcripts، تغيير normalizer) بتعمل version جديدة؛ أدوات زي DVC أو LakeFS أو حتى manifests immutable في object store مع naming منضبط كفاية.

**سؤال متابعة:** الـ GPU utilization في التدريب 40%. ايه الأسباب المحتملة في الـ data pipeline وازاي تشخّصها؟ (profiling للـ dataloader، عدد الـ workers، الـ decoding، الـ augmentation، الـ IO للـ shards).

### س114. الـ splits والـ leakage والـ imbalance: ليه الـ random split غلط في الكلام، وازاي تكتشف التسريب، وازاي تتعامل مع لهجة عندها 500 ساعة وأخرى 20 ساعة؟

**الإجابة النموذجية:**
الـ random split على مستوى الـ utterances بيحط نفس المتكلم في الـ train والـ test، فالموديل بيتعلم صوته والأرقام بتبقى متفائلة وما بتتحققش في الـ production مع متكلمين جداد. القاعدة: speaker-disjoint splits، ولو الـ data مقروءة من نصوص، text-disjoint كمان (نفس الجملة بأصوات مختلفة في الـ train والـ test بتخلي الموديل "يحفظ" النص). وفي المكالمات، session-disjoint (نفس المكالمة ما تتقسمش). ولو الـ data من قنوات ومصادر، الـ test لازم يشمل مصادر مش في الـ train عشان تقيس التعميم.

اكتشاف التسريب: مقارنة speaker embeddings بين الـ train والـ test (أزواج بتشابه عالي = نفس الشخص حتى لو الـ IDs مختلفة)، ومقارنة n-grams النص، وhash الصوت (والـ near-duplicate بالـ fingerprinting لأن نفس الملف ممكن يكون بـ encoding مختلف). وعلامة عملية: لو الـ test WER أقل بكتير من الـ WER على data جديد من الـ production، فيه تسريب أو الـ test مش ممثل.

الـ imbalance: الموديل هيتعلم اللهجة الكبيرة ويهمل الصغيرة. الحلول: sampling بيرفع اللهجة الصغيرة (temperature sampling بحيث ما يزيدش تكرارها لدرجة الـ overfitting؛ 20 ساعة مكررة 25 مرة بتتحفظ)، وaugmentation أقوى للصغيرة، وcurriculum: تدريب عام ثم fine-tuning قصير مع خلط، وpseudo-labeling لصوت غير معنون من اللهجة الصغيرة، والأهم القياس بالـ slices: الـ WER لكل لهجة منفصل، والقرار بناءً على الأسوأ مش المتوسط. وتقبل إن 20 ساعة مش هتوصل لنفس الجودة، فتخطط لجمع data بدل ما تحاول تحل بالـ tricks.

**سؤال متابعة:** لهجة عندها data كتير بس كلها من 10 متكلمين. ايه المشكلة وايه الحل؟ (الموديل بيتعلم المتكلمين مش اللهجة؛ الحل تنويع المتكلمين ولو بساعات أقل، وvoice conversion كحل مؤقت).

### س115. جمع speech data بالـ crowdsourcing (قراءة جمل أو محادثات) في السعودية: تصميم النصوص، تنوع الأجهزة والمتكلمين، الـ validation، الدفع والأخلاقيات. وامتى الـ crowdsourcing ما ينفعش؟

**الإجابة النموذجية:**
الـ crowdsourcing بيشتغل كويس للـ read speech (جمل محددة) وللـ prompted speech (رد على سؤال بحرية) عبر تطبيق موبايل. التصميم: (1) النصوص: للقراءة تختار جمل قصيرة متنوعة phonetically وdomain-wise بلغة قريبة من الكلام (مش نصوص صحفية جافة) وتشمل أرقام وأسماء، ولليهجات تكتب الجمل باللهجة أو تدي سيناريو ("اطلب من الخدمة تغيير رقم جوالك") عشان الكلام يطلع عفوي. (2) التنوع: تحدد quotas للمناطق والجنس والعمر ونوع الجهاز، والتسجيل بميكروفون الموبايل في بيئات مختلفة (بيت، شارع، سيارة) عمدًا لأن ده الـ production. (3) الـ validation: خطوة ثانية في التطبيق: متكلمين تانيين بيسمعوا التسجيل ويتحققوا إنه مطابق للنص وواضح (زي Common Voice)، زائد فلترة آلية (VAD، SNR، الطول، STT مقارنة). (4) الدفع: عادل ومعلن، والمهمة قصيرة (10-15 دقيقة)، وحوافز للتنوع مش للكمية بس. (5) الأخلاقيات والقانون: موافقة صريحة موضحة لاستخدام الصوت (والـ cloning لو مطلوب)، حق الحذف، عدم جمع بيانات زيادة عن الحاجة، وتخزين داخل المملكة.

امتى ما ينفعش: (1) المكالمات الحقيقية العفوية بين طرفين (لا تُصنّع بالـ crowdsourcing؛ محتاجة تسجيل من مركز اتصال حقيقي أو محادثات مُدارة بين اتنين بسيناريوهات). (2) الـ TTS data (محتاجة متكلم محترف واستوديو). (3) الـ domain الحساس (بيانات بنكية حقيقية). (4) لما محتاج جودة نسخ عالية: القراءة بتدي النص مجانًا، لكن الكلام الحر محتاج annotation بعدين.

النتيجة الواقعية: مئات الساعات من الـ read/prompted speech في أسابيع بتكلفة معقولة، وده بيغطي التنوع الصوتي والأجهزة، وبيفضل محتاج مكالمات حقيقية للـ conversational patterns.

**سؤال متابعة:** الـ contributors بيقروا الجمل بنبرة "قراءة" رسمية حتى لو النص باللهجة. ازاي تحفّز الكلام الطبيعي؟

### س116. الـ data لموديلات الـ turn-taking والـ endpointing والـ barge-in: ايه اللي بتحتاجه، وازاي تعنونه، وليه الـ ASR corpora العادية ما تنفعش؟

**الإجابة النموذجية:**
موديل الـ turn detection محتاج يتعلم "المتكلم خلّص ولا لسه هيكمل"، وده بيتعلم من محادثات حقيقية بين طرفين بقنوات منفصلة، لأن الـ ASR corpora فيها utterances مقصوصة أصلًا عند حدود واضحة، فمفيش فيها الحالات الصعبة: الوقفة في نص الجملة للتفكير، الجملة اللي بتنتهي بنغمة معلقة، الـ fillers ("يعني"، "امم")، والمقاطعات.

الـ data: مكالمات مركز اتصال بقناتين (أو محادثات مسجلة) بتتقسم لـ turns، ولكل نقطة صمت أطول من 100-200 ms بتتعنون: هل بعدها نفس المتكلم كمّل (pause داخل الـ turn) ولا الطرف التاني بدأ (turn end)؟ الـ labels دي بتتولّد آليًا من الـ timing بين القناتين، وده اللي بيخلي الموضوع قابل للتوسع: آلاف المكالمات بدون annotation يدوي، مع تنظيف للحالات الغامضة (الاتنين اتكلموا مع بعض). الـ features: النص من الـ STT للـ turn الحالي (زي turn detector بتاع LiveKit اللي بيشتغل على النص)، وأحيانًا الـ audio (النغمة في آخر 500 ms) زي Smart Turn، والأفضل الاتنين.

للـ barge-in: محتاج أمثلة لمقاطعات حقيقية مقابل backchannels ("أيوه"، "تمام" أثناء كلام الـ agent مش مقاطعة)، وده بيتعنون من مكالمات فيها agent بيتكلم (بشري أو آلي) والعميل بيرد أثناءه، والـ label: هل المتكلم عايز يوقف الطرف التاني ولا بيأكد بس.

اللهجات هنا بتفرق جدًا: الـ fillers والنغمات الختامية مختلفة بين المصري والسعودي، والـ turn detector المتدرب على إنجليزي بيقطع المتكلم العربي في نص الجملة، فلازم data محلي. والتقييم: precision/recall عند نقاط الصمت، ومقياس مباشر على الـ latency (متوسط التأخير بعد نهاية الـ turn الحقيقية) ومعدل القطع الخاطئ.

**سؤال متابعة:** الـ turn detector النصي بيحتاج transcript، والـ transcript بيتأخر. ازاي تتعامل مع الـ latency دي في التصميم؟ (يشتغل على الـ partial، ويتحد مع الـ VAD بحيث لو الصمت طال بيقرر من غيره).

---

## 8. الأمان والخصوصية والـ safety

### س117. الـ speaker verification والـ anti-spoofing للمصادقة بالصوت: ازاي بيشتغل، الـ EER، والتعامل مع الـ deepfakes.

**الإجابة النموذجية:**
**الـ speaker verification:** استخراج embedding للمتكلم (x-vector ثم ECAPA-TDNN ثم WavLM/ResNet-based) من الـ enrollment audio ومن المكالمة، ومقارنة بـ cosine similarity أو PLDA، وقرار بـ threshold. text-independent (أي كلام) أسهل للمستخدم لكن أضعف، text-dependent (عبارة ثابتة) أقوى. المقاييس: FAR (قبول محتال) وFRR (رفض صاحب الحساب)، والـ EER هي النقطة اللي بيتساووا فيها، لكن في البنوك بتضبط الـ threshold عند FAR واطي جدًا وتقبل FRR أعلى. الـ channel mismatch (تسجيل من app والمصادقة من تليفون) بيرفع الأخطاء وبيحتاج data من القناتين.

**الـ anti-spoofing والـ liveness:** الهجمات: replay (تسجيل صوت العميل)، TTS وvoice conversion (وده بقى سهل جدًا بالـ zero-shot cloning). الدفاع: موديلات spoof detection (AASIST وأشباهها المدربة على ASVspoof) بتلتقط artifacts الـ synthesis، وliveness بعبارة عشوائية (يقرأ رقم مختلف كل مرة فالـ replay ما ينفعش)، وwatermark detection للأصوات المولّدة من أنظمتك أنت، وتحليل الـ channel.

**الواقع الحالي:** مع جودة الـ cloning، الصوت لوحده ما يصلحش كعامل مصادقة وحيد لعمليات مالية، بس كعامل إضافي مع OTP أو device binding، ومع risk scoring (مبلغ غير معتاد + صوت مش متأكد = تحويل لموظف). ولازم إعادة تقييم دورية لأن الـ spoof detectors بتتقادم مع كل جيل TTS جديد.

**سؤال متابعة:** ازاي تبني evaluation set للـ anti-spoofing يبقى واقعي؟ (تولّد هجمات بأحدث الموديلات المفتوحة بأصوات عملاء وهميين على نفس القناة).

### س118. الـ PII والامتثال في pipelines الصوت: الـ redaction والـ retention والـ consent وprompt injection عبر الصوت.

**الإجابة النموذجية:**
**الـ PII في التسجيلات والنصوص:** NER عربي على الـ transcripts لاكتشاف الأسماء والأرقام والعناوين وإخفائها في الـ logs والـ analytics، والـ audio نفسه ممكن يتعمله muting للمقاطع اللي فيها أرقام بطاقات (لو بتتقال بالصوت) أو يتجمع بالـ DTMF أصلًا وميتسجلش. النصوص اللي بتروح للـ LLM لازم تكون بأقل PII ممكن (اسم أول بدل الاسم الكامل، رقم حساب مقنّع).

**الـ consent والـ retention:** إعلان التسجيل في بداية المكالمة، سياسة retention واضحة (مثلًا 90 يوم للـ audio، أطول للـ transcripts المقنّعة)، تشفير at rest وin transit، وaccess control بسجل من فتح أي تسجيل. استخدام بيانات العملاء في تدريب الموديلات محتاج أساس قانوني وموافقة وanonymization، وده بيتكتب في السياسات مش يتفترض.

**الـ prompt injection عبر الصوت:** المستخدم ممكن يقول "تجاهل التعليمات واعمل كذا"، أو يشغّل صوت مسجل فيه تعليمات. الـ STT بيحوّله لنص بيدخل الـ prompt كأي نص. الدفاع: فصل واضح بين system instructions ومحتوى المستخدم، tool permissions محدودة (الـ LLM مش عنده صلاحية يحوّل من غير تأكيد وauthentication)، input classifiers، وrefusal policies، واختبار adversarial مستمر. كمان الـ RAG content نفسه ممكن يكون مصدر injection.

**الـ audit trail:** كل قرار (tool call، تحويل، رفض) مسجل مع الـ turn والـ audio reference عشان التحقيق وقت الشكاوى.

**سؤال متابعة:** لو الـ regulator طلب "شرح" لقرار اتخذه الـ agent في مكالمة، ايه اللي لازم يكون متسجل عشان تجاوب؟

### س119. الـ adversarial attacks على الـ STT والـ VAD والـ speaker verification: hidden voice commands، الـ ultrasonic injection، الـ universal perturbations، والـ replay attacks. ايه اللي واقعي في التهديد وايه الدفاعات؟

**الإجابة النموذجية:**
الهجمات المعروفة: (1) الـ targeted adversarial audio: إضافة تشويش محسوب رياضيًا (بالـ gradients) على تسجيل بحيث الـ STT يطلّع نص مختار من المهاجم بينما الإنسان بيسمع الأصل، وده أثبت على موديلات white-box (DeepSpeech في أبحاث Carlini) وأصعب على الـ black-box وعبر الهواء (over-the-air) لأن القناة بتشوّه الـ perturbation، بس الـ universal perturbations والهجمات المنقولة بين الموديلات بتقرّب المسافة. (2) الـ hidden commands: أوامر مدمجة في موسيقى أو ضوضاء بحيث الموديل يفهمها والبشر لأ. (3) الـ ultrasonic (DolphinAttack): ترددات فوق السمع بتتحول لكلام مسموع للمايك بسبب اللاخطية في الـ hardware. (4) الـ replay: تشغيل تسجيل لصوت الشخص أمام نظام الـ speaker verification. (5) هجمات على الـ VAD والـ endpointing (ضوضاء تمنع الـ endpoint فالـ agent ما يردش) كـ DoS بسيط.

التقييم الواقعي: في voice agent تليفوني، الهجوم الأخطر عمليًا هو الـ social engineering والـ replay والـ deepfake على التحقق الصوتي، مش الـ adversarial perturbation المعقدة. لكن في الأجهزة الاستهلاكية (سماعات ذكية، سيارات) الـ hidden commands والـ ultrasonic تهديد حقيقي.

الدفاعات: (1) عدم الاعتماد على الصوت لوحده لأي إجراء حساس (تأكيد، عامل ثاني). (2) الـ adversarial training والـ input transformations (resampling، compression، إضافة noise) اللي بتكسر الـ perturbations الدقيقة بثمن بسيط في الدقة. (3) فلتر low-pass عند الـ hardware ضد الـ ultrasonic. (4) الـ liveness detection والـ anti-spoofing (كشف الـ replay من خصائص السماعة والقناة) والـ random passphrase. (5) الـ anomaly detection: نص مفهوم من صوت "ضوضاء" حسب الـ VAD إشارة للهجوم. (6) الـ rate limiting وقيود على طول المكالمة ومحاولات التحقق. والمرشح القوي بيوصل إن الـ threat model يتكتب أولًا: مين المهاجم، ايه اللي يكسبه، وايه القناة، وبعدين تختار الدفاعات المناسبة بدل ما تحاول تدافع ضد كل الأبحاث.

**سؤال متابعة:** الـ adversarial training بيقلل الدقة على الـ clean speech. ازاي تقرر الـ tradeoff في نظام بنكي؟

### س120. الـ privacy-preserving speech: الـ speaker anonymization، المعالجة على الجهاز، الـ data minimization، الـ federated learning، والموافقة على التسجيل في السياق السعودي. ازاي تبني نظام صوتي "privacy by design"؟

**الإجابة النموذجية:**
الصوت بيحمل الهوية (بصمة صوتية) والمحتوى والحالة الصحية والعاطفية أحيانًا، فهو بيانات شخصية حساسة. المبادئ: (1) الـ data minimization: ما تخزنش الصوت إلا لو محتاجه فعلًا، وتخزن الـ transcript المطبّع بدل الصوت لما يكفي، وتحدد retention قصير للصوت الخام (أيام لأسابيع) وأطول للـ transcripts المنقّحة. (2) الـ redaction عند المصدر: كتم المقاطع الحساسة (أرقام، أسماء) في الصوت والنص قبل التخزين، بالـ NER على الـ transcript مع timestamps والـ DTMF detection. (3) المعالجة على الجهاز للـ wake word والأوامر البسيطة بحيث الصوت ما يسيبش الجهاز. (4) الـ speaker anonymization (VoicePrivacy): تغيير الصوت بـ voice conversion قبل مشاركة data مع vendors أو للبحث، مع فهم إن المحتوى نفسه ممكن يكشف الهوية وإن الـ anonymization بتقلل جودة الـ data للـ TTS. (5) الـ federated learning مناسب نظريًا للـ wake word وpersonalization على الأجهزة، لكنه معقد تشغيليًا وقليل الاستخدام في الـ enterprise voice، والـ differential privacy على نماذج الكلام لسه بحثي في الغالب.

في السياق السعودي: الـ PDPL بيتطلب أساس قانوني وإشعار وغرض محدد، فالمكالمة لازم تبدأ بإعلان التسجيل والغرض، والموافقة على استخدام الصوت للتدريب لازم تكون منفصلة وواضحة ("لتحسين الخدمة" مش كافية غالبًا)، وحقوق الأفراد (الوصول، التصحيح، الحذف) لازم لها إجراء فعلي، والـ data residency داخل المملكة، ومع البنوك متطلبات SAMA الإضافية للتخزين والوصول. وتشغيل أي API خارجي للـ STT/TTS بيبقى نقل بيانات لخارج المملكة وده غالبًا غير مقبول للبنوك والجهات الحكومية، وده أحد أهم أسباب الـ on-prem.

المرشح القوي بيذكر الـ privacy impact assessment كوثيقة تتعمل قبل التصميم، والـ access control والـ audit للسماع (مين سمع أي مكالمة وليه)، والفصل بين بيئة التدريب وبيئة الـ production.

**سؤال متابعة:** فريق الـ ML عايز يسمع مكالمات حقيقية عشان يحسّن الموديل. ايه الإجراء الآمن؟ (عينة محدودة، بعد redaction، في بيئة معزولة، بموافقات موثقة وسجل وصول).

### س121. نشر deepfake detection في نظام تحقق صوتي: مشكلة التعميم على مولّدات جديدة، الـ false positives على عملاء حقيقيين، الـ watermark verification، وبروتوكول التقييم.

**الإجابة النموذجية:**
موديلات الـ anti-spoofing (AASIST، RawNet2، وموديلات مبنية على WavLM/wav2vec features) بتدي EER منخفض على الـ benchmarks (ASVspoof) لكن المشكلة الأساسية التعميم: الموديل بيتعلم artifacts المولّدات اللي شافها، ولما يظهر TTS جديد (وده بيحصل كل شهر) الأداء بينهار. والمشكلة التانية الـ false positives: العميل الحقيقي على تليفون رديء أو بـ codec غريب أو بصوت مبحوح بيتصنّف "مزيف"، وده أخطر تجاريًا من تمرير هجوم نادر.

التصميم العملي: (1) الـ detector طبقة من طبقات، مش قرار نهائي: نتيجته بتتجمع مع الـ speaker verification والـ liveness (random passphrase أو رد تفاعلي) والـ risk signals (رقم المتصل، سلوك الحساب، حجم العملية)، وفي الحالات المشكوك فيها بيتم التصعيد لعامل تاني (OTP) مش الرفض. (2) التدريب المستمر: كل ما يظهر TTS/VC جديد بتولّد بيه data وتعيد التدريب، وبتحتفظ بمكتبة من المولّدات، مع augmentation بالـ codecs والقنوات عشان الـ artifacts ما تتخبيش. (3) الـ watermark: لو الـ TTS بتاعك (أو بتاع شريك) بيحط watermark (AudioSeal أو مشابه)، الـ detector بيتحقق منه بسرعة وبثقة، بس ده بيمسك مولّداتك أنت بس، والمهاجم بيستخدم مولّد تاني، فالـ watermark مفيد للـ provenance والامتثال مش للدفاع. (4) الـ provenance standards (C2PA للصوت) بتساعد لما المحتوى جاي من مصادر موقّعة.

التقييم: test set من مولّدات مش في التدريب (held-out generators) هو الرقم الحقيقي، وقياس الـ FPR على آلاف المكالمات الحقيقية المتنوعة بالقنوات الرديئة، ومراقبة في الـ production لنسبة التصنيف "مزيف" عبر الوقت (قفزة مفاجئة يا هجوم يا مشكلة في القناة). وتقرير واضح للـ business إن الكشف احتمالي ومش ضمانة، وإن الصوت مش عامل تحقق وحيد.

**سؤال متابعة:** عايز تقيس كام هجوم deepfake حقيقي بيحصل على النظام. ازاي وأنت مش عارف الـ ground truth؟

### س122. أمان سلسلة توريد الموديلات في الـ on-prem: تحميل weights مفتوحة المصدر بأمان، الـ pickle مقابل safetensors، التحقق من الـ hashes والتراخيص، الـ SBOM، والتحديث في بيئة معزولة.

**الإجابة النموذجية:**
الموديلات المفتوحة بتيجي كملفات وأكواد من مصادر خارجية، وده سطح هجوم حقيقي: (1) الـ pickle (.pt/.pth/.bin) بينفّذ كود عند التحميل، فملف موديل ممكن يشغّل أي أمر؛ القاعدة: safetensors فقط للـ weights، وأي checkpoint بصيغة pickle يتحوّل في بيئة معزولة (sandbox بدون شبكة) قبل ما يدخل الشبكة الداخلية، وتحميل بـ weights_only لما يكون لازم. (2) الـ trust_remote_code في HuggingFace بيشغّل كود من الـ repo، فلازم مراجعة الكود المحمّل وتثبيته بـ commit hash مش بـ branch. (3) التحقق: hash (SHA-256) لكل ملف مقارنة بالمصدر الرسمي، والتحميل من الـ organization الرسمية مش من mirrors، وتخزين نسخة داخلية موقّعة في registry خاص (model registry مع signatures). (4) التراخيص: لكل موديل ملف license محفوظ مع الـ weights وتوثيق القرار التجاري، والانتباه للـ licenses اللي بتتغير بين الإصدارات (نفس الاسم بترخيص مختلف). (5) الـ dependencies: مكتبات معالجة الصوت (ffmpeg، libsndfile، decoders) لها تاريخ ثغرات في parsing الملفات، وملف صوت خبيث من مستخدم ممكن يستغلها، فالتحديث الدوري والـ sandboxing لمرحلة الـ decoding وحدود على حجم وصيغة الملفات المقبولة. (6) الـ SBOM (قائمة كل المكونات والإصدارات) لكل deployment، والـ scanning للـ images. (7) في البيئة المعزولة: قناة تحديث رسمية (جهاز نقل معتمد)، وتحقق من الـ hashes بعد النقل، وrollback محفوظ.

كمان الأكواد اللي بتتولّد من الموديلات نفسها (زي prompts أو configs) ما بتنفّذش تلقائيًا، وحسابات الخدمات بأقل صلاحيات، ومفاتيح التشفير للـ weights لو الموديل ملكية خاصة.

**سؤال متابعة:** ليه ما ينفعش تعتمد على "الـ repo مشهور وله نجوم كتير" كدليل أمان؟ (سرقة الحسابات وتبديل الملفات حصلت فعلًا في الـ ecosystem).

### س123. اكتب threat model لـ voice agent بنكي: سطوح الهجوم (التليفون، التحقق، الحوار، الـ tools، الـ data)، والهجمات المتوقعة لكل واحد، والتخفيفات ذات الأولوية.

**الإجابة النموذجية:**
سطوح الهجوم والهجمات: (1) التليفون: caller ID spoofing (رقم العميل بيتزوّر بسهولة على SIP)، فما ينفعش يكون عامل تحقق؛ وهجمات DoS بمكالمات كتير طويلة بتستهلك الـ GPUs. (2) التحقق: replay وdeepfake على الـ voice biometrics، وsocial engineering للـ agent عشان يتخطى خطوات التحقق ("أنا بنت العميل"). (3) الحوار: prompt injection (تعليمات في كلام المستخدم أو من طرف تالت على الخط)، واستخراج معلومات بالتدرج (سؤال عن الرصيد بطرق مختلفة)، ودفع الـ agent لوعود أو معلومات غلط (التزام قانوني). (4) الـ tools: تنفيذ عمليات بصلاحيات أوسع من اللازم، وtool calls بمعاملات متلاعب فيها (تحويل لحساب المهاجم)، وعدم الـ idempotency (تكرار تحويل). (5) الـ data: تسرب من الـ logs والتسجيلات، ووصول داخلي غير مصرح، ونقل خارج المملكة عبر API. (6) الـ TTS: الـ agent يقرا بصوت عالي بيانات لشخص جنب العميل (speaker phone)، والصوت المسجل يستخدم لتدريب مولّد لتقليد صوت الشركة في هجمات على العملاء (vishing).

التخفيفات بالأولوية: (1) الصلاحيات: الـ agent يقدر يعمل بس العمليات منخفضة الخطر بدون تحقق قوي، والعمليات المالية بـ OTP أو تحقق متعدد، والـ tools بحدود مبلغ وقوائم مسموحة، وكل tool مع side effect إما idempotent أو محمي بـ confirmation. (2) فصل التعليمات عن المحتوى المسموع، وقواعد ثابتة خارج الـ LLM (policy engine) للأشياء اللي ما ينفعش الـ LLM يقررها. (3) rate limits على المكالمات والمحاولات والتحقق. (4) عدم قراءة بيانات حساسة كاملة بالصوت (آخر 4 أرقام). (5) تسجيل وتدقيق كامل مع redaction، وإنذار على الأنماط الشاذة (مكالمات كتير من نفس الرقم لعملاء مختلفين). (6) الـ anti-spoofing كطبقة إضافية للتحقق الصوتي. (7) watermark على صوت الـ agent وتوعية العملاء إن البنك ما بيطلبش بيانات بالطريقة دي. (8) red teaming دوري بسيناريوهات صوتية باللهجات المحلية، وتعامل مع النتائج كـ bugs بأولوية.

**سؤال متابعة:** الـ business عايز الـ agent يعمل تحويلات بدون OTP عشان "تجربة سلسة". ازاي تعرض المخاطر وتقترح بديل؟

---

## 9. الـ debugging والخبرة العملية

### س124. المستخدمين بيشتكوا إن الـ agent "بطيء". ازاي تشخّص المشكلة؟

**الإجابة النموذجية:**
المرشح القوي ما يبدأش بتخمين، يبدأ بالـ instrumentation: timestamp موحد لكل turn في نقاط: نهاية كلام المستخدم (من الـ audio نفسه مش من الـ VAD event)، قرار الـ endpoint، الـ STT final، أول token من الـ LLM، آخر token، أول byte من الـ TTS، وأول audio اتشغّل عند الـ client (مش اتبعت). من غير الرصد ده كل النقاش نظري.

بعد كده: (1) شوف الـ p95 مش الـ average، وقسّم حسب طول المكالمة (لو الـ latency بتزيد مع الوقت = الـ context بيكبر والـ TTFT بيزيد). (2) الـ endpointing threshold غالبًا المتهم الأول، وأحيانًا الـ VAD بيتأخر بسبب noise. (3) الـ GPU queueing تحت الضغط (الـ latency كويسة الصبح وسيئة في الذروة). (4) الـ tools: استعلام بطيء من نظام خارجي. (5) الشبكة: RTT بين الـ services أو TURN relay للـ client. (6) الـ TTS chunks طويلة أوي فأول جملة بتتأخر. (7) الـ client buffer كبير.

وفي النهاية يعيد إنتاج المشكلة بمكالمات synthetic تحت الحمل، ويحط SLO وalert على كل مرحلة عشان المشكلة ما ترجعش من غير ما تتشاف.

**سؤال متابعة:** ازاي تفرّق بين "بطيء" و"بيقطع المستخدم" في الشكاوى؟ الاتنين بيتوصفوا بنفس الكلام أحيانًا.

### س125. احكيلي عن نظام صوتي بنيته ووصل production: ايه اللي كسر، ايه المقاييس، وايه اللي كنت هتعمله مختلف؟

**اللي بتدور عليه في الإجابة (مش إجابة نموذجية):**
- أرقام حقيقية: WER قبل وبعد، latency p95، containment، عدد المكالمات. المرشح اللي ما عندوش أرقام غالبًا ما كانش قريب من الـ production.
- تشخيص واضح لمشكلة حقيقية: hallucination على الصمت، endpointing بيقطع الناس، echo على التليفون، أرقام بتتفهم غلط، موديل اتدهور بعد تحديث.
- حلول data-centric مش بس model-centric: غيّر الـ guidelines، جمع data من القناة الصح، عمل regression set.
- وعي بالـ tradeoffs: ضحى بدقة عشان latency ولا العكس، وليه.
- ownership: on-call، rollback، تواصل مع الـ product.
- صراحة: "ده ما اشتغلش" أو "كنت هبدأ بـ X بدل Y" علامة نضج.

**علامات سلبية:** كلام عام عن "طبّقنا Whisper وكان كويس"، عدم معرفة الـ metrics، تحميل كل المشاكل على الـ data أو على فريق تاني، وعدم ذكر أي evaluation.

### س126. الـ WER زاد من 12% لـ 18% بعد deploy لموديل الـ test بتاعه كان أحسن. امشي معايا في التشخيص المنهجي.

**الإجابة النموذجية:**
الفرضية الأولى دايمًا: مش الموديل، الـ pipeline حواليه. الترتيب: (1) هل الـ 18% متقاسة بنفس الطريقة؟ نفس الـ normalizer ونفس الـ test set ونفس الـ scoring؟ فرق في الـ normalization (تشكيل، همزة، أرقام) بيعمل 5% بسهولة. (2) الـ preprocessing في الـ serving: sample rate الداخل فعلًا (هل الـ PBX بيبعت 8k والكود بيفترض 16k؟)، الـ resampling library، الـ int16/float scaling، الـ mel parameters والـ normalization constants، الـ channel (stereo اتحول mono بطريقة مختلفة). طريقة التحقق: تاخد 100 ملف من الـ production وتشغّلهم offline بالـ pipeline بتاعة الـ test؛ لو النتيجة 12% يبقى المشكلة في الـ serving path، ولو 18% يبقى الـ data اختلف. (3) الـ VAD/segmentation في الـ production بيقطع مختلف عن الـ test (segments أقصر أو بتقطع في نص الكلمة). (4) الـ tokenizer/vocab أو ملف الـ config اللي اتحمّل مع الـ weights مش المطابق (نسخة قديمة من الـ normalizer أو الـ ITN). (5) الـ export: fp16 overflow، TensorRT profile، فرق في الـ decoding (greedy في الـ serving وbeam+LM في الـ test). (6) لو كل ده سليم: الـ distribution في الـ production مختلف (لهجة، ضوضاء، domain جديد بسبب حملة) والـ test set مش ممثل، وهنا بتبني test set جديد من الـ production.

الأداة الأساسية: golden set صغير من الـ production بتشغّله على كل مرحلة (raw audio، بعد الـ preprocessing، بعد الموديل، بعد الـ postprocessing) وتقارن بالمرجع، وdiff للـ configs بين الـ staging والـ production. والمرشح القوي بيقول إن الـ rollback الفوري أول قرار، والتشخيص بعده.

**سؤال متابعة:** طلع إن الـ WER على الـ production كان 18% من قبل الـ deploy كمان، بس محدش كان بيقيسه. ايه اللي يتغير في العملية؟

### س127. صوت الـ TTS في الـ production بقى "مختلف": أكثر روبوتية، أو pitch أعلى، أو فيه clicks. الموديل ما اتغيرش. ايه الأسباب المحتملة بالترتيب؟

**الإجابة النموذجية:**
لما الـ weights ثابتة والصوت اتغير، المتهمين: (1) الـ sample rate mismatch: الموديل بيطلّع 24 kHz والـ player أو الـ telephony gateway بيتعامل معاه كـ 22.05 أو 16 kHz (الصوت أعلى pitch وأسرع، أو أوطى وأبطأ)، وده بيحصل بعد تحديث في الـ media server أو تغيير header. (2) الـ chunk boundaries: تغيير في الـ chunking أو الـ crossfade بيعمل clicks عند الوصلات، أو الـ client بيشغّل الـ chunks بـ gaps. (3) الـ quantization أو الـ export: نسخة جديدة من الـ inference engine اتعملها fp16 وبوّظت الـ vocoder، أو تغيير في الـ ONNX runtime. (4) الـ sampling parameters: temperature أو CFG أو seed أو عدد الـ NFE اتغيروا في config جديد. (5) الـ reference/prompt audio للـ zero-shot models اتغير أو اتقص أو اتعمله resampling غلط. (6) الـ text frontend: normalizer أو diacritizer جديد بيطلّع نص مختلف (التشكيل أو الوقف)، أو lexicon اتعدّل. (7) الـ loudness normalization أو codec في الـ telephony path اتغير. (8) الـ GPU/driver: kernel numerics مختلفة (نادر لكن بيحصل مع bf16 على hardware جديد).

التشخيص: تولّد نفس الجملة بنفس الـ seed على الـ staging والـ production وتقارن الـ waveforms (لو متطابقين المشكلة بعد الموديل في الـ playback path)، وتشغّل الملف الناتج مباشرة بدون النظام، وتشوف الـ spectrogram (pitch shift واضح بيدل على sample rate، وclicks بتدل على boundaries). والدرس: golden audio set بيتولّد ويتقارن آليًا (بـ mel distance وspeaker similarity) في كل deploy، لأن الأذن البشرية بتلاحظ بس بعد ما العملاء يشتكوا.

**سؤال متابعة:** الـ mel distance بين الـ staging والـ production صفر، والعملاء لسه بيشتكوا إن الصوت مختلف. فين تدوّر؟ (الـ telephony path والـ codecs والـ loudness، أو تغيّر في النصوص اللي الـ LLM بيولّدها فالـ prosody اختلف).

### س128. شكاوى "الـ agent بيقاطعني" و"الـ agent مش بيسمعني". ازاي تفصل بين المشكلتين وتشخّص كل واحدة من الـ logs والتسجيلات؟

**الإجابة النموذجية:**
الشكوتين متشابهتين لكن الأسباب عكس بعض. "بيقاطعني": الـ endpoint بيتقرر بدري: min silence قصير، أو الـ turn detector بيعتبر الوقفة نهاية، أو الـ VAD بيسقط الكلام الواطي فبيشوف صمت، أو الـ LLM بيرد على partial. التشخيص: من التسجيل بقناتين تحسب لكل turn الفرق بين آخر كلام حقيقي للمستخدم وبداية صوت الـ agent؛ لو الـ agent بدأ قبل ما المستخدم يخلّص (تداخل) في نسبة كبيرة من الـ turns فده تأكيد، وتصنّف الحالات: وقفة تفكير، جملة طويلة، أرقام (الناس بتقف بين المجموعات)، ضوضاء غطّت الكلام. "مش بيسمعني": الـ VAD مش بيفتح (threshold عالي، صوت واطي، AGC ناقص)، أو الـ AEC بيقمع صوت المستخدم أثناء كلام الـ agent (فبيبان إن الـ barge-in مش شغال)، أو الـ endpoint مش بيتقرر (ضوضاء مستمرة بتخلي الـ VAD مفتوح فالـ agent مستني للأبد)، أو الـ STT بيطلّع فاضي على كلام حقيقي. التشخيص: لكل مكالمة مشتكية تعرض الـ timeline: الـ VAD state، الـ partials، الـ endpoint events، فوق الـ waveform الحقيقي، وتشوف فين الفجوة.

الحلول مختلفة: للمقاطعة تطوّل الصمت المطلوب وتحسّن الـ turn detector وتعمل معالجة خاصة للأرقام؛ لـ "مش بيسمعني" تظبط الـ VAD وتضيف AGC وتراجع الـ AEC وتضيف حد أقصى للانتظار مع سؤال ("لسه معايا؟"). والمقياسان اللي لازم يكونوا في الـ dashboard: overlap rate (نسبة الـ turns اللي الـ agent بدأ فيها قبل نهاية كلام المستخدم) وno-response rate (نسبة الـ turns اللي المستخدم اتكلم فيها ومفيش رد خلال ثانيتين).

**سؤال متابعة:** الشكاوى كلها من مستخدمي iPhone على شبكة معينة. ايه اللي ممكن يكون مختلف؟ (الـ codec والـ jitter وسلوك AEC الجهاز، أو الـ AGC بتاع الشبكة).

### س129. الـ demo على اللابتوب ممتاز، وأول مكالمات حقيقية على التليفون كارثة. اعمل checklist للفروق اللي لازم تتحضر ليها قبل أي pilot تليفوني.

**الإجابة النموذجية:**
الفروق الجوهرية بين الـ demo والتليفون: (1) الصوت: 8 kHz وG.711 بدل 48 kHz Opus، فالـ STT يفقد الـ fricatives، والـ TTS يطلع مكتوم؛ الحل موديل مدرّب/مقيّم على telephony data وtest set تليفوني. (2) الـ AEC: على اللابتوب المتصفح بيعمل AEC، وعلى التليفون مفيش، فالـ barge-in بيتشغّل على صوت الـ agent نفسه (echo) أو ما بيشتغلش؛ محتاج AEC على الـ media server أو منطق يتجاهل الـ VAD أثناء كلام الـ agent إلا لو الطاقة عالية. (3) الـ latency: الشبكة التليفونية بتضيف 100-300 ms في كل اتجاه، وjitter، فالـ budget بيتأكل. (4) الـ DTMF والـ early media ونغمات الانتظار والـ hold: الـ VAD بيفتح على نغمات، والـ STT بيهلوس على موسيقى الانتظار. (5) المستخدمين الحقيقيين: بيتكلموا قبل ما التحية تخلّص، بيسكتوا، بيقولوا الأرقام بسرعة، بيتكلموا من الشارع، وناس تانية بتتكلم جنبهم؛ الـ demo كان بيتكلم فيه مهندس واضح في مكتب هادي. (6) اللهجة والـ code-switching بنسب أكبر من الـ test set. (7) الـ concurrency: الـ demo مكالمة واحدة، والـ pilot عشرات، فالـ latency تحت الحمل مختلفة. (8) الـ caller ID والـ routing والـ transfer مع الـ PBX الحقيقي. (9) الـ compliance: إعلان التسجيل والإفصاح.

الـ checklist العملية قبل الـ pilot: اختبار على خط تليفوني حقيقي بـ 8k من أجهزة مختلفة، test set تليفوني للـ STT، AEC وbarge-in policy مختبرة على echo حقيقي، معالجة الـ DTMF والـ hold music، load test بـ 50% من الـ concurrency المتوقعة، fallback للموظف بضغطة، recording وdashboard شغالين من أول مكالمة، وفريق بيسمع المكالمات الأولى بنفسه في نفس اليوم.

**سؤال متابعة:** أول 20 مكالمة حقيقية: 8 منهم قفلوا في أول 10 ثواني. ايه الفرضيات وازاي تتحقق منها من التسجيلات؟

### س130. مشاكل التدريب الشائعة في موديلات الكلام: الـ loss بيطلع NaN، الـ OOM مع الـ utterances الطويلة، الـ WER بيثبت بدري، والموديل بيتعلم يطلّع blank بس. ايه الأسباب والحلول؟

**الإجابة النموذجية:**
(1) NaN/Inf: غالبًا fp16 overflow في الـ attention logits أو في الـ CTC/RNN-T loss (الـ log-space computation)، أو learning rate عالي من غير warmup كافي، أو utterances فيها صوت صفر/DC أو features غير متناهية بسبب log(0). الحل: bf16 بدل fp16 (أو loss scaling)، gradient clipping، warmup أطول، فحص الـ inputs، وتخطي الـ batches اللي فيها non-finite loss مع logging. (2) OOM: الـ utterances الطويلة بتفجّر الـ attention (تربيعي) والـ transducer joint (T×U×V)؛ الحل حد أقصى للطول (20-30 ثانية) وتقطيع الأطول، dynamic batching بحسب إجمالي الثواني، pruned RNN-T، gradient checkpointing، وترتيب الـ buckets بحيث أطول batch يتجرّب أولًا عشان الـ OOM يظهر في أول دقيقة مش بعد يوم. (3) الـ WER بيثبت بدري: learning rate واطي أو schedule بينزل بسرعة، الموديل صغير على الـ data، الـ augmentation قوي جدًا، label noise عالي (الموديل مش قادر يتعلم من transcripts غلط)، أو الـ tokenizer غير مناسب؛ التشخيص بمقارنة train loss وdev WER: لو الـ train loss بينزل والـ dev ثابت فـ overfitting/leakage أو الـ dev مختلف عن الـ train في التوزيع. (4) الموديل بيطلّع blank بس (CTC collapse) في أول التدريب: عادي لفترة قصيرة، لكن لو استمر: learning rate عالي، subsampling كبير جدًا بالنسبة لطول النص (القيد T أكبر من U مكسور)، أو الـ transcripts أطول من الصوت (ملفات مقطوعة)، أو الـ blank weight/initialization. (5) الـ attention model بيتعلم alignment غلط: CTC auxiliary loss بيحل.

والمبدأ العام: ابدأ بموديل صغير على subset صغير وتأكد إنه بيحفظه (overfit) قبل أي تدريب كبير، وراقب الـ alignment plots والـ gradient norms، وسجّل كل حاجة في experiment tracker.

**سؤال متابعة:** الـ train WER 3% والـ dev WER 25% من أول epoch. ايه أول حاجة تشك فيها؟ (تسريب بين train وdev، أو الـ dev من توزيع مختلف تمامًا).

### س131. ورقة بحثية بتدّعي WER أقل بـ 30% على العربي بمعمارية جديدة. ازاي تقيّمها قبل ما تستثمر شهر في إعادة إنتاجها؟

**الإجابة النموذجية:**
الأسئلة بالترتيب: (1) المقارنة عادلة؟ نفس الـ training data وحجم الموديل والـ compute والـ decoding (beam+LM ضد greedy بتعمل الفرق ده لوحدها)، ونفس الـ normalization في الـ scoring؟ أغلب "التحسينات الكبيرة" بتختفي لما الـ baseline يتظبط. (2) الـ test set: معروف ومقفول ولا من اختيارهم؟ فيه احتمال contamination؟ الأرقام على أكتر من set؟ (3) الـ ablation: هل الورقة بتبين أي مكوّن عمل الفرق، ولا التحسين جاي من data أكتر أو من tricks معروفة (SpecAugment، checkpoint averaging)؟ (4) الكود والـ weights متاحين؟ لو لا، الاحتمال إنك تعيد إنتاج الرقم بيقل كتير. (5) التكلفة: الـ inference أغلى؟ streaming ممكن؟ export ممكن؟ تحسين 30% بموديل مش قادر يشتغل real-time مش مفيد للـ agent. (6) مين الناس؟ فرق لها سجل في الإنتاج بتنشر أرقام أقرب للواقع من أوراق مبنية على benchmark واحد.

الخطة: تجربة صغيرة محدودة الوقت (يومين): تشغيل الـ weights المنشورة لو موجودة على test set بتاعك (مش بتاعهم)، ومقارنة بـ baseline بتاعك بنفس الـ decoding. لو الفرق موجود على data بتاعتك حتى لو أقل، يستاهل الاستثمار؛ لو اختفى، تكتب ملاحظة قصيرة وتتحرك. والمرشح القوي بيذكر إن أغلب المكاسب الحقيقية في العربي جاية من الـ data والـ normalization والـ decoding مش من المعمارية، وإن قراءة الـ appendix والـ hyperparameters أهم من قراءة الـ abstract.

**سؤال متابعة:** الورقة بتقارن على MGB-2 بس. ايه اللي MGB-2 ما بيقيسوش بالنسبة لمنتجك؟ (بث إعلامي MSA نظيف، مش تليفون ولا لهجات عفوية).

### س132. عندك ربع سنة لتحسين الـ STT للـ voice agent، والفريق 4 مهندسين. ازاي تحدد الأولويات وتقول لا للاقتراحات اللي مش هتفرق (زي "نجرب أحدث موديل" كل أسبوع)؟

**الإجابة النموذجية:**
البداية بالقياس مش بالأفكار: أسبوع لبناء error analysis من الـ production: توزيع الأخطاء حسب النوع (أرقام، أسماء، لهجة، ضوضاء، code-switch، hallucination) والتأثير على الـ task (أنهي أخطاء بتفشّل المكالمة فعلًا). غالبًا بتلاقي إن 60% من الفشل من 2-3 أسباب محددة. بعدها بترتّب المبادرات بـ (التأثير المتوقع × الاحتمال) / التكلفة: مثلًا (1) تصليح الـ normalization والـ ITN للأرقام: أسبوع ومكسب مباشر. (2) جمع 100 ساعة من مكالمات حقيقية باللهجة الأكثر فشلًا وfine-tuning: 6 أسابيع، مكسب كبير محتمل. (3) الـ contextual biasing للأسماء والمنتجات: أسبوعين. (4) تحسين الـ endpointing: مكسب في الـ latency وتقليل القطع. أما "تجربة أحدث موديل" فبتتحط في مسار evaluation ثابت: أي موديل جديد بيتقيّم في يوم واحد على الـ test set بـ pipeline جاهز، ولو ما كسبش الـ baseline على data بتاعتك ما بيدخلش نقاش، وده بيحوّل الحماس لعملية بدل ما يبقى جدل.

الـ "لا" بتتقال بالأرقام: "الموديل ده على test set بتاعنا 14% مقابل 12% الحالي، والتكلفة أعلى، فمش هنغيّر"، وبتحافظ على 10-15% من وقت الفريق للاستكشاف عشان ما تقتلش الفضول. وفي نهاية الربع الـ report بيربط كل مبادرة بالـ metric اللي اتحسن وبالتأثير على الـ containment، وده اللي بيبني الثقة مع الإدارة. والمرشح القوي بيذكر إن أهم قرار غالبًا هو الاستثمار في الـ data والـ evaluation infrastructure لأنه بيرفع كل حاجة بعده.

**سؤال متابعة:** الإدارة عايزة رقم "الـ WER هيبقى كام في نهاية الربع". ازاي تجاوب من غير ما تعد بحاجة مش مضمونة؟

### س133. حصل outage في الـ voice agent لمدة ساعة في الذروة. ازاي تدير الحادث لحظيًا، وازاي تعمل postmortem مفيد بدون لوم؟

**الإجابة النموذجية:**
لحظيًا: (1) أول قرار تخفيف الأثر مش فهم السبب: تحويل المكالمات للـ IVR التقليدي أو الطابور البشري عبر الـ fallback route المجهز مسبقًا، وإعلان داخلي واضح (قناة incident، مين الـ incident commander، تحديث كل 15 دقيقة). (2) rollback لأي تغيير حديث (deploy، config، تحديث في الـ PBX) قبل التحليل العميق. (3) التشخيص بالـ dashboards: أنهي مرحلة كسرت أولًا (الـ timeline بيبين الـ error rate ابتدى فين)، وهل المشكلة في المكوّن ولا في dependency (الشبكة، الـ storage، الـ LLM server، الـ SIP trunk). (4) الاحتفاظ بالأدلة (logs، metrics snapshots) قبل الـ restart.

الـ postmortem خلال أيام: timeline دقيق بالدقايق، الأثر بالأرقام (مكالمات فشلت، عملاء اتأثروا)، السبب الجذري والأسباب المساهمة (مثلًا: تحديث في الـ codec في الـ PBX خلّى الـ STT يستقبل صوت غلط، والـ alert على الـ confidence ما كانش موجود، والـ fallback كان بطيء لأن محدش جربه من 6 شهور)، وليه ما اتكشفش بدري، وأسئلة "ايه اللي كان هيمنعه" و"ايه اللي كان هيخلي الاكتشاف أسرع" و"ايه اللي كان هيخلي التعافي أسرع". الـ action items محددة بأسماء ومواعيد، بأولوية للاكتشاف والتعافي مش بس المنع، وبدون لوم شخصي: السؤال "ايه في النظام سمح بده" مش "مين غلط".

المرشح القوي بيذكر تدريبات دورية على الـ fallback (game days)، وإن أغلب الـ outages في الأنظمة الصوتية أصلها في الـ integration (PBX، شبكة، تحديثات خارجية) مش في الموديلات، وإن الـ SLO والـ error budget بيديروا نقاش "سرعة الـ features مقابل الاستقرار" بشكل موضوعي.

**سؤال متابعة:** الـ postmortem طلّع 15 action item. ازاي تختار اللي هيتنفذ فعلًا؟

---

## 10. تدريب foundation speech models من الصفر

الأسئلة دي للمرشحين اللي هيشتغلوا على تصميم وتدريب موديلات الكلام نفسها (STT وTTS وcodecs وSSL) مش على استخدامها بس. المطلوب هنا خبرة فعلية في training runs كبيرة: الـ data pipelines، الـ distributed training، الاستقرار، والقرارات اللي بتتاخد تحت قيود الـ compute.

### س134. امتى تدرّب موديل STT عربي من الصفر، وامتى تكتفي بـ continued pretraining أو fine-tuning لموديل موجود؟ ايه معايير القرار؟

**الإجابة النموذجية:**
المعايير: (1) حجم الـ data: fine-tuning بيكفي مع مئات الساعات، والـ continued pretraining على SSL model يستاهل مع آلاف الساعات غير معنونة، والتدريب من الصفر بيبدأ يبقى منطقي مع عشرات الآلاف من الساعات المعنونة (أو مئات الآلاف مع pseudo-labels) وإلا هتطلع أضعف من Whisper أو Parakeet على كل حاجة. (2) الفجوة في الموديلات الموجودة: لو الموديلات المتاحة ضعيفة في اللهجات لدرجة إن الـ fine-tuning مش بيوصل للهدف (لأن الـ encoder نفسه ما شافش الأصوات دي كفاية)، الـ continued pretraining أو التدريب من الصفر بيفرقوا. (3) التراخيص والملكية: منتج بيتباع للحكومة والبنوك ممكن يحتاج موديل ملكيته وبياناته معروفة بالكامل، وده بيدفع للتدريب الذاتي. (4) الـ architecture constraints: لو محتاج streaming حقيقي بـ latency واطية وexport سهل، الموديلات الـ CTC/TDT بتاعتك أنسب من fine-tuning Whisper. (5) الـ compute والفريق: التدريب من الصفر مشروع شهور بفريق وGPUs، والـ fine-tuning أيام.

المسار الواقعي لأغلب الفرق: يبدأ بـ fine-tuning لأحسن موديل متاح (Whisper large-v3 أو FastConformer) لتحديد الـ baseline والـ error analysis، وبالتوازي يبني الـ data pipeline والـ evaluation، ثم continued pretraining لـ encoder (w2v-BERT 2.0 أو XLS-R أو BEST-RQ بتاعك) على كل الصوت العربي المتاح (عشرات الآلاف من الساعات غير معنونة أسهل بكتير في الجمع)، ثم تدريب CTC/transducer عليه بالـ data المعنونة والـ pseudo-labels. التدريب الكامل من الصفر بيبقى المرحلة الأخيرة لما الـ data والفريق يكونوا جاهزين، ومقارنته بالـ baseline على نفس الـ test sets هي اللي بتبرره.

**سؤال متابعة:** ايه الحجة ضد "Whisper fine-tuned كفاية"؟ ومتى تكون الحجة دي صح فعلًا؟

### س135. خطط training run لموديل STT عربي بحجم مليار parameter تقريبًا على 100 ألف ساعة. المعمارية، الـ data، تقدير الـ compute (بالأرقام)، الجدول الزمني، والمخاطر.

**الإجابة النموذجية:**
المعمارية: encoder Conformer/FastConformer أو E-Branchformer بـ 24 layer وd_model 1024 (حوالي 600M-1B)، مع head CTC وhead transducer (أو AED) للتدريب المشترك، 8x subsampling، BPE عربي-إنجليزي 1024-2048 token. الـ data: 100 ألف ساعة مقسمة بين معنون بشري (الأعلى وزنًا)، pseudo-labeled بموديل قوي بعد فلترة، وread speech، بتوزيع لهجات محسوب، مع dev/test مقفولين.

تقدير الـ compute: الـ encoder بيعالج frames بعد الـ subsampling. عند 80 ms لكل frame: 100 ألف ساعة = 3.6×10^8 ثانية = 4.5×10^9 frame. تقريب الـ FLOPs للتدريب حوالي 6 × params × frames لكل epoch (مع تجاهل الـ attention التربيعي اللي بيضيف نسبة): 6 × 10^9 × 4.5×10^9 ≈ 2.7×10^19 FLOPs لكل epoch على الـ encoder. لو الـ GPU بيدي فعليًا 3×10^14 FLOP/s (حوالي 30-40% MFU من H100 bf16، وده واقعي مع data loading والـ augmentation)، الـ epoch على 8 GPUs ≈ 2.7×10^19 / (8 × 3×10^14) ≈ 1.1×10^4 ثانية ≈ 3 ساعات. مع الـ speed perturbation (3x) و5-10 epochs ≈ 50-100 ساعة GPU-node، يعني أيام على node واحدة. الرقم ده بيبين إن الـ compute مش الـ bottleneck الحقيقي للـ STT بالحجم ده؛ الـ bottleneck هو الـ data pipeline (decoding وaugmentation لـ 100 ألف ساعة بيحتاج عشرات الـ CPU cores لكل GPU) والـ IO والتجارب المتكررة. (المرشح مش لازم يطلّع نفس الأرقام، المهم المنهج والوعي إن الـ decoder/attention والـ data loading بيغيروا الصورة).

الجدول: 4-6 أسابيع تجهيز data وpipeline وbaseline صغير، أسبوعين تجارب على 10% من الـ data بموديل 100M لاختيار الـ hyperparameters، ثم الـ run الكبير أسبوع مع مراقبة، ثم أسبوعين evaluation وexport. المخاطر: label noise، تسريب للـ test، انهيار في نص الـ run (checkpointing كل ساعة)، الـ data pipeline أبطأ من الـ GPU، وأخيرًا اكتشاف إن الموديل أضعف من الـ baseline في لهجة معينة بسبب توزيع الـ data.

**سؤال متابعة:** لو الموديل AED بـ decoder كبير (زي Whisper)، ازاي بيتغير تقدير الـ compute؟ (الـ decoder بيعالج tokens أقل بكتير من الـ frames فتكلفته أقل، لكن الـ cross-attention والـ inference هما اللي بيغلوا).

### س136. الـ self-supervised pretraining من الصفر للعربي: wav2vec 2.0 مقابل HuBERT مقابل BEST-RQ. الـ objectives، مشاكل الاستقرار، وليه BEST-RQ بقى الخيار العملي؟ وايه recipe الـ continued pretraining؟

**الإجابة النموذجية:**
wav2vec 2.0 بيعمل contrastive learning: بيقنّع أجزاء من الـ latent features وبيخلي الموديل يميّز الـ quantized target الصح من distractors، مع diversity loss عشان الـ codebook ما ينهارش. مشاكله: حساس للـ hyperparameters (temperature الـ Gumbel، وزن الـ diversity)، وبيحصل codebook collapse، وبيحتاج batches كبيرة جدًا. HuBERT بيعمل masked prediction لـ targets من k-means على features (MFCC في أول iteration ثم layers الموديل نفسه)، أثبت وأسهل، بس بيحتاج iterations (تدريب، clustering، تدريب تاني) وde-facto بيحتاج pipeline للـ clustering على data ضخم. BEST-RQ (اللي بني عليه USM من Google) بسّط كل ده: الـ targets من random projection ثابت وcodebook عشوائي مجمّد على الـ mel features، والموديل بيتوقع الـ codes للأجزاء المقنّعة؛ مفيش quantizer يتعلم فمفيش collapse، ومفيش iterations، وبيشتغل مباشرة على mel، وبيوصل لنتائج مماثلة أو أحسن على نطاق كبير. عشان كده بقى الخيار العملي لفريق محدود.

الـ recipe للـ continued pretraining على اللهجات (الحالة الأكثر شيوعًا): تبدأ من w2v-BERT 2.0 أو XLS-R أو MMS، وتجمع 10-50 ألف ساعة صوت عربي غير معنون متنوع (مكالمات بعد الموافقة، بودكاست مرخّص، تسجيلات داخلية)، تفلتر بالـ VAD والجودة الأساسية، وتكمل التدريب بنفس الـ objective وlearning rate أقل من الأصلي (5-10 مرات) لمئات الآلاف من الـ steps، مع خلط نسبة من data متنوعة اللغات عشان ما تكسرش التمثيلات، وتقيس التقدم بـ probe: fine-tuning CTC سريع على 100 ساعة معنونة كل فترة والمقارنة بالـ WER على dev لهجي. الخطأ الشائع إن الـ loss بينزل والـ downstream ما بيتحسنش، وده بيحصل لما الـ data غير معنونة ضيقة أو الـ LR عالي.

**سؤال متابعة:** ليه الـ SSL features بتفيد الـ TTS وspeaker tasks كمان مش بس الـ STT، وأنهي layers بتفيد كل مهمة؟

### س137. الـ distributed training لموديلات الكلام: DDP مقابل FSDP، الـ mixed precision، الـ dynamic batching عبر GPUs، الـ gradient accumulation، الـ checkpointing، وقياس الـ throughput. ايه اللي بيختلف عن تدريب LLM نصي؟

**الإجابة النموذجية:**
لحد مليار parameter الـ DDP كافي (كل GPU عنده نسخة كاملة والـ gradients بتتجمع بالـ all-reduce)، وفوق كده أو مع optimizer states كبيرة FSDP/ZeRO بيقسّم الـ parameters والـ gradients والـ optimizer states بين الـ GPUs. الـ mixed precision: bf16 على Ampere/Hopper هو الافتراضي (مفيش loss scaling ومشاكل overflow أقل)، مع الاحتفاظ بالـ loss computation (CTC/RNN-T) في fp32. الاختلافات عن الـ LLM: (1) الـ batches غير متجانسة الطول جدًا، فبتعمل dynamic batching بحد أقصى لإجمالي الثواني لكل GPU (مثلًا 300-600 ثانية)، وده معناه إن عدد الـ utterances بيختلف بين الـ GPUs وبين الـ steps، ولازم الـ sampler يضمن إن كل الـ GPUs بتاخد أحمال متقاربة وإلا الأسرع بتستنى الأبطأ (straggler). (2) الـ data loading أثقل: decoding صوت وaugmentation على CPU لكل sample، فبتحتاج workers كتير وprefetch، وعادة GPU utilization بيكون أقل من اللغة. (3) الـ sequence lengths متغيرة فالـ memory متغيرة، والـ OOM بيظهر في batch نادر طويل؛ الحل حدود صارمة على الطول وترتيب الـ buckets. (4) الـ gradient accumulation بيوصّل لـ effective batch كبير (ساعات صوت في الـ step الواحد بتحسّن الاستقرار) لما الـ GPUs محدودة. (5) الـ checkpointing: كل ساعة تقريبًا مع الاحتفاظ بآخر N، والـ resume لازم يرجّع حالة الـ sampler والـ RNG عشان ما تعيدش نفس الـ data، والـ checkpoint averaging لآخر checkpoints بيدي مكسب مجاني في الـ WER.

قياس الـ throughput: ساعات صوت لكل ساعة GPU (أو frames/sec) وMFU، والمقارنة بالمتوقع؛ لو الـ GPU utilization أقل من 60% ابدأ بالـ profiler على الـ dataloader. والمرشح القوي بيذكر أدوات: NeMo وESPnet وicefall وLhotse بيدوا كل ده جاهز، والكتابة من الصفر مبررة بس لبحث معماري.

**سؤال متابعة:** الـ training على 8 GPUs بيدي 5x بس مش 8x مقارنة بـ GPU واحد. ايه الأسباب المحتملة؟ (stragglers من الـ dynamic batching، الـ dataloader، الـ all-reduce على شبكة بطيئة، أو batch صغير لكل GPU).

### س138. الاستقرار والـ hyperparameters في تدريب موديلات الكلام الكبيرة: الـ LR schedule، حجم الـ batch بالثواني، الـ warmup، الـ gradient clipping، الـ loss spikes، والـ EMA والـ checkpoint averaging.

**الإجابة النموذجية:**
الـ Transformers الصوتية حساسة في البداية: من غير warmup كافي الـ attention بيتشتت والـ loss بيقفز. الـ recipes المعتادة: Noam schedule (زيادة خطية لحد peak عند 10-25k step ثم انخفاض بجذر الـ step) في ESPnet، أو warmup ثم cosine/linear decay في NeMo وWhisper، مع peak LR بيقل مع كبر الموديل (حوالي 1e-3 للصغير و1e-4 أو أقل للكبير مع AdamW). حجم الـ batch بيتقاس بالثواني: من 20 دقيقة لساعات من الصوت لكل step للموديلات الكبيرة، وكل ما كبر كل ما اللر يقدر يعلى والتدريب يستقر. الـ gradient clipping عند 1.0 شبه إلزامي، والـ weight decay صغير (0.01-0.1)، والـ dropout 0.1 مع layerdrop اختياري.

الـ loss spikes: بتظهر كقفزة مفاجئة في الـ loss مع أو بدون تعافي. الأسباب: batch فيه utterance شاذة (ضوضاء بحتة مع transcript، أو طول غير طبيعي)، أو LR عالي في مرحلة معينة، أو precision. التعامل: كشف الـ spike آليًا (loss أعلى من متوسط متحرك بمقدار كبير) وتخطي الـ step، ولو تكرر ترجع لآخر checkpoint وتقلل الـ LR أو تغيّر الـ seed، وتفحص الـ batch اللي عمل الـ spike (سجّل الـ ids). الـ EMA للـ weights (متوسط متحرك أسّي) بيدي موديل أنعم وأفضل في الـ evaluation وبيتستخدم في TTS كتير، والـ checkpoint averaging (متوسط آخر 5-10 checkpoints) بيدي نفس الأثر تقريبًا في الـ STT بدون تكلفة أثناء التدريب.

المنهج: sweep على موديل صغير وdata subset لاختيار الـ LR والـ warmup، ثم تحويل الـ settings للكبير بحذر (اللر ينزل مع الحجم)، مع run قصير على الكبير قبل الـ run الكامل عشان تشوف أول 5k step. والمرشح القوي بيذكر إن التدريب "المستقر" مش معناه الأفضل: اللر الواطي جدًا مستقر وبيدي موديل أضعف.

**سؤال متابعة:** الـ loss بيقل بسلاسة بس الـ dev WER بيتذبذب بقوة بين الـ checkpoints. ايه التفسير وايه الحل؟ (الـ dev صغير أو الموديل في مرحلة LR عالي؛ dev أكبر، وaveraging، وتقييم بـ beam ثابت).

### س139. الـ curriculum والـ data mixing في الـ pretraining: من القصير للطويل، من النظيف للـ noisy، الـ temperature sampling للهجات، عدد الـ epochs مقابل الـ data الفريدة، وامتى توقف التدريب.

**الإجابة النموذجية:**
الـ curriculum بيساعد الاستقرار في البداية: تبدأ بـ utterances قصيرة ونظيفة نسبيًا (أو تحد الطول الأقصى في أول epoch) ثم تفتح الطول والـ augmentation تدريجيًا، لأن الـ attention بيتعلم الـ alignment أسهل على القصير. الـ mixing: كل مصدر له وزن (بشري معنون أعلى من pseudo-labeled، وread speech أقل من conversational لو الهدف مكالمات)، واللهجات بـ temperature sampling عشان الصغيرة تتمثل من غير ما تتكرر لدرجة الحفظ، والإنجليزي أو اللغات الأخرى بنسبة صغيرة لو محتاج code-switching أو transfer. الـ pseudo-labels بتتضاف بعد ما الموديل الأول يبقى قوي (noisy student) مع فلترة بالـ confidence، وبتتعاد كل دورة بموديل أحسن.

الـ epochs: الموديلات الصوتية بتستفيد من تكرار الـ data مع augmentation (الـ augmentation بيعمل "data جديد" جزئيًا)، فـ 5-10 epochs على data كبير عادي، لكن مع data صغير التكرار الكتير بيحفظ. القاعدة: data فريد أكتر دايمًا أحسن من epochs أكتر، والـ pseudo-labeling طريقة رخيصة لزيادة الفريد. الإيقاف: لما الـ dev WER على الـ slices المهمة يثبت لفترة (patience بعدة evaluations) أو لما الـ compute budget ينتهي، مع الأخذ في الاعتبار إن الـ LR decay في النهاية بيدي مكسب أخير، فبتخطط الـ schedule على عدد steps محدد بدل early stopping العشوائي.

المرشح القوي بيذكر الـ dedup قبل حساب الـ epochs (الـ data المكرر بيخلي "epoch واحد" فعليًا 3 epochs على بعض الجمل)، وإن الـ mixing weights بتتحدد بتجارب صغيرة، وإن تغيير الـ mix في نص الـ run (زي إضافة data جديد) لازم يتعمل بحذر مع إعادة warmup صغيرة.

**سؤال متابعة:** أضفت 20 ألف ساعة pseudo-labeled والـ WER اتحسن في المتوسط بس ساء على الأرقام والأسماء. ليه؟ (الموديل المعلم بيغلط فيها، فالـ pseudo-labels بتثبّت الغلط؛ الحل فلترة خاصة أو استبعاد الجمل اللي فيها أرقام من الـ pseudo-labels).

### س140. تدريب neural audio codec من الصفر: المعمارية، الـ losses (reconstruction وadversarial وcommitment)، الـ RVQ tricks ضد الـ collapse، الـ semantic distillation، الـ data، والتقييم. وايه القرارات اللي بتحدد ملاءمته للـ TTS مقابل الـ speech LLM؟

**الإجابة النموذجية:**
المعمارية: encoder conv (SEANet style) بيوصل من الـ waveform لـ latent بـ frame rate منخفض (50-75 Hz تقليديًا، و12.5-25 Hz في الحديث بـ downsampling أكبر أو transformer bottleneck)، ثم quantizer (RVQ بعدة codebooks، أو FSQ، أو single codebook كبير)، ثم decoder mirror. الـ losses: (1) reconstruction على الـ waveform (L1) وعلى multi-scale mel/STFT (بأحجام windows مختلفة) عشان تلقط التفاصيل الزمنية والترددية. (2) adversarial بـ multi-scale STFT discriminators وMPD (زي DAC) مع feature matching، وده اللي بيدي الوضوح والطبيعية عند bitrates واطية. (3) commitment loss للـ VQ عشان الـ encoder outputs تلزم الـ codes، مع EMA update للـ codebooks. الـ tricks ضد الـ collapse: k-means initialization للـ codebooks، استبدال الـ codes الميتة (dead code replacement) بـ vectors من الـ batch، factorized codes بأبعاد صغيرة وL2-normalized lookup (DAC)، أو FSQ اللي بيلغي المشكلة أصلًا، وquantizer dropout عشان الموديل يشتغل بعدد codebooks متغير (bitrate scalable).

الـ semantic distillation: عشان أول codebook يحمل معلومات لغوية (أسهل للـ LLM)، بتضيف loss يقرّب الـ latent الأول من features WavLM أو HuBERT (SpeechTokenizer وMimi)، أو تدخل encoder semantic منفصل (X-codec). ده بيحسّن الـ speech LLMs على حساب شوية جودة صوتية عند نفس الـ bitrate.

الـ data: عشرات الآلاف من الساعات من كلام نظيف ومتنوع عند 16/24 kHz، مع نسبة من الضوضاء والموسيقى لو الـ codec عام؛ والـ segments قصيرة (ثانية أو اتنين) في التدريب. التقييم: mel/STFT distance، ViSQOL أو PESQ/STOI، SI-SNR، WER بـ ASR على الصوت المعاد بناؤه (الوضوح)، speaker similarity، codebook usage/perplexity، وMOS على عينة، كل ده مقابل الـ bitrate.

قرارات الملاءمة: للـ TTS عالي الجودة بتفضّل frame rate وbitrate أعلى (جودة صوتية)، وللـ speech LLM بتفضّل frame rate واطي (12.5 Hz) وcodebook أول semantic وعدد codebooks قليل عشان الـ sequence قصيرة والـ LM يتعلم بسهولة، والـ multi-scale (SNAC) حل وسط.

**سؤال متابعة:** ليه الـ codec المتدرب على الإنجليزي بيشتغل معقول على العربي، وامتى بيفشل؟ (الصوت البشري عام، لكن الأصوات الحلقية والمفخمة وتوزيع الـ F0 ممكن يبقوا outliers في الـ codebook).

### س141. تدريب LLM-style TTS من الصفر (CosyVoice أو F5 style) للعربي: الـ data pipeline من الصوت الخام، الـ tokenizer، مراحل التدريب، استراتيجية التشكيل في التدريب، والـ compute.

**الإجابة النموذجية:**
الـ data pipeline (على نمط Emilia-Pipe): جمع صوت مرخّص بكميات كبيرة (بودكاست مرخّص، كتب صوتية، تسجيلات داخلية، read speech من الـ crowdsourcing)، توحيد الصيغة، source separation لإزالة الموسيقى، diarization لفصل المتكلمين والاحتفاظ بالـ segments أحادية المتكلم، VAD وتقطيع لـ 3-20 ثانية، ASR بموديل قوي للـ transcripts (وهنا جودة الـ Arabic ASR بتاعك بتحدد جودة الـ TTS)، فلترة بالـ DNSMOS والـ language consistency وطول النص مقابل الصوت، وdedup للمتكلمين. النتيجة عادة 20-40% من الخام. للعربي بتضيف: تحقق إن النص ما فيهش تشكيل عشوائي من الـ ASR، وتوحيد الهمزات والتاء المربوطة.

الـ tokenizer الصوتي: إما تدرّب supervised semantic tokenizer (encoder ASR مع VQ في المنتصف، زي S3 في CosyVoice) على data ASR بتاعتك، أو codec. النصي: BPE عام (من الـ LLM) أو حروف. مراحل التدريب: (1) الـ tokenizer/codec. (2) الـ LM (نص → semantic tokens) على كل الـ data، مبدئيًا من LLM نصي صغير (0.5-1.5B) عشان يبدأ بمعرفة لغوية. (3) الـ flow matching decoder (tokens + speaker embedding → mel) على نفس الـ data أو subset أنظف. (4) الـ vocoder (أو استخدام جاهز زي BigVGAN/HiFT). (5) SFT على أعلى جودة (استوديو) وأصوات الـ brand. (6) اختياريًا RL (DPO أو reward من ASR WER وspeaker similarity) لتقليل الـ hallucination.

استراتيجية التشكيل: ثلاث خيارات: تدريب على نص بدون تشكيل (الموديل يتعلم النطق من السياق، بيحتاج data ضخم وبيغلط في الـ homographs)، أو تدريب على نص مشكّل آليًا (بيحتاج diacritizer في الـ inference وبيورّث أخطاءه)، أو الهجين: تشكّل الـ data بـ diacritizer جيد ثم في التدريب تسقط التشكيل عشوائيًا (كليًا أو جزئيًا) فالموديل يقبل النص بالحالتين ويستفيد من التشكيل لما يكون موجود، وده بيخلي الـ lexicon والـ override ممكنين بالتشكيل الصريح. الـ compute: الـ LM على 50-100 ألف ساعة (حوالي 2-5 مليار semantic token عند 25 Hz) بموديل 0.5-1B هو أيام على 8 GPUs، والـ flow decoder مماثل؛ الـ pipeline والفلترة والتجارب هما الوقت الحقيقي.

**سؤال متابعة:** الموديل بيتكلم MSA ممتاز وبيقرا النص اللهجي بنطق فصيح غريب. ايه السبب في الـ data وازاي تحل؟ (الـ data اللهجي قليل أو الـ ASR كتبه بصيغة فصحى؛ الحل data لهجي حقيقي بـ transcripts لهجية، وdialect tag).

### س142. التقييم أثناء الـ pretraining: ايه الـ dev sets والـ proxy metrics، وامتى الـ WER على dev صغير بيضلل، وازاي تختار الـ checkpoint، وازاي تستخدم موديلات صغيرة للـ ablations؟

**الإجابة النموذجية:**
أثناء الـ run الكبير، الـ evaluation لازم يكون رخيص ومتكرر: dev sets صغيرة (30 دقيقة لكل slice: MSA، لهجات، تليفون، noisy) بتتقيّم كل بضع آلاف من الـ steps بـ greedy decoding (مش beam) عشان السرعة، مع الـ loss على dev كـ proxy أول. الـ proxy metrics بتضلل في حالات: (1) الـ dev صغير فالتذبذب أكبر من الفرق الحقيقي. (2) الـ greedy WER ما بيعكسش الـ WER النهائي بالـ LM. (3) الـ dev من نفس توزيع الـ train (بيتحسن باستمرار بينما الـ production لأ). (4) في الـ SSL، الـ pretraining loss ما بيتنبأش بالـ downstream، فلازم probe fine-tuning دوري. (5) في الـ TTS، الـ loss ما له علاقة قوية بالجودة المسموعة، فبتعمل توليد لعينة ثابتة من الجمل كل فترة وتحسب WER بالـ ASR وspeaker similarity وUTMOS وتسمع بنفسك.

اختيار الـ checkpoint: مش آخر واحد بالضرورة؛ الـ averaging لآخر N ثم التقييم الكامل (beam + LM) على dev كبير، ثم الـ test مرة واحدة في الآخر. والـ ablations: تعملها على موديل صغير (50-100M) وsubset (5-10% من الـ data) عشان تجاوب أسئلة زي "الـ tokenizer الأكبر أحسن؟" أو "الـ augmentation X بيفيد؟" في ساعات بدل أيام، مع الحذر إن بعض النتائج ما بتتنقلش للكبير (الموديل الكبير بيتحمل noise أكتر ومحتاج regularization أقل)، فالـ ablations المهمة بتتأكد على scale متوسط قبل القرار النهائي. وكل تجربة بتتسجل بـ config كامل ونتيجة على نفس الـ dev، وإلا بعد شهر محدش فاكر ليه اتاخد القرار.

**سؤال متابعة:** الـ dev WER اتحسن 0.3% بعد تغيير. ازاي تعرف إن ده حقيقي؟ (تكرار بـ seeds مختلفة، وbootstrap على dev أكبر).

### س143. ايه اللي معروف عن الـ scaling في موديلات الكلام (Whisper وOWSM وUSM وSeamless): الـ data مقابل الـ parameters، الـ diminishing returns، والـ interference بين اللغات؟ وازاي تستخدم تجارب صغيرة للتنبؤ بالكبير؟

**الإجابة النموذجية:**
الملاحظات المنشورة: (1) Whisper بيّن إن زيادة الـ data (من عشرات الآلاف لمئات الآلاف من الساعات) بتستمر تحسّن الأداء الـ multilingual والـ robustness، بينما الإنجليزي بيوصل لـ saturation بدري، وإن تكبير الموديل مع data ثابت بيدي مكاسب متناقصة، وإن الموديلات الكبيرة أكتر robustness للـ distribution shift. (2) OWSM (إعادة إنتاج مفتوحة بـ 180 ألف ساعة data عامة) وصل قريب من Whisper وبيّن إن الفرق بيرجع للـ data مش للسر المعماري، وإن E-Branchformer أحسن من Transformer عادي عند نفس الحجم. (3) USM (Google) بيّن قوة الـ SSL pretraining على ملايين الساعات غير المعنونة (BEST-RQ) ثم fine-tuning بـ data معنونة أقل بكتير، وإن الـ pretraining بيقلل الحاجة للـ labels بشكل كبير للغات قليلة الموارد. (4) الـ interference: مع موديل صغير، إضافة لغات بتضر اللغات الكبيرة، ومع موديل كبير الـ transfer إيجابي غالبًا؛ والعربي بيستفيد من اللغات ذات الـ script والأصوات القريبة.

الاستخدام العملي: تعمل scaling curve بنفسك: تدرب 3-4 موديلات بأحجام مختلفة (30M، 100M، 300M) على نسب مختلفة من الـ data (10%، 30%، 100%) وترسم الـ WER مقابل الـ compute، فتقدر تقدّر مكسب المرحلة الجاية وتقرر هل الاستثمار في data أكتر ولا موديل أكبر، وتلاقي غالبًا إن للعربي اللهجي الـ data هي المتغير الأهم بفارق كبير. والمرشح القوي بيحذر إن الـ scaling laws في الكلام أقل نضجًا من النص، وإن جودة الـ labels بتكسر أي منحنى.

**سؤال متابعة:** الفريق عايز يدرّب موديل 3B "عشان أكبر أحسن". ايه الحجة المضادة بالأرقام من scaling curve؟

### س144. تدريب موديل STT يشتغل streaming وnon-streaming بنفس الـ weights: الـ dynamic chunk training، الـ causal convolutions، الـ left context، والـ latency-accuracy curve. ايه القرارات في التدريب؟

**الإجابة النموذجية:**
الفكرة (من WeNet U2/U2++ وNeMo cache-aware FastConformer): تدرّب الـ encoder بـ attention masks بتحد الـ context حسب chunk size، وفي كل batch تختار chunk size عشوائي (من 160 ms لـ full context)، فالموديل الواحد يقدر يشتغل بأي chunk في الـ inference: chunk صغير للـ real-time وfull للـ offline بنفس الـ weights. الـ convolutions لازم تبقى causal (أو بـ lookahead محدود) وإلا بتسرّب من المستقبل، والـ BatchNorm بيتبدّل بـ LayerNorm، والـ positional encoding relative عشان يشتغل مع الـ chunks. الـ left context: الموديل بيشوف عدد محدد من الـ chunks السابقة (مثلًا 5-10 ثواني) بدل الكل عشان الـ compute والـ memory ثابتين مع طول المكالمة، وده بيقلل الدقة شوية بس بيسمح بمكالمات بلا نهاية. NeMo بيدرب على مجموعة من الـ attention context sizes (multiple lookaheads) فتقدر تختار وقت الـ inference من عدة نقاط على منحنى الـ latency-accuracy.

القرارات: (1) توزيع الـ chunk sizes في التدريب (لو full-context كتير، الـ streaming بيبقى أضعف والعكس). (2) هل الـ decoder (attention) كمان streaming ولا CTC بس للـ streaming والـ attention للـ rescoring في الآخر (U2 بيعمل كده: CTC للـ partials وattention rescoring للـ final). (3) الـ subsampling rate: 8x بيقلل الـ compute بس بيخلي أصغر chunk 80 ms. (4) الـ right context/lookahead: كل 100 ms lookahead بتزود الدقة وتزود الـ latency. (5) التقييم بمنحنى كامل: WER عند 160/320/640/1280 ms وfull، وemission latency لكل نقطة، والاختيار حسب الـ budget بتاع الـ agent.

للعربي مفيش خصوصية معمارية هنا، لكن الـ dialect data مع chunk صغير بيكون أصعب لأن الكلمة الطويلة الملتصقة (بادئات ولواحق) محتاجة context، فالفجوة بين streaming وoffline ممكن تبقى أكبر من الإنجليزي ولازم تتقاس.

**سؤال متابعة:** ليه الـ streaming model بيتأخر في إطلاق الكلمة (emission latency) حتى مع chunk صغير، وازاي التدريب يقلل ده؟ (الـ CTC بيأجل الـ token لحد ما يتأكد؛ tricks زي peak-first regularization أو الـ delay penalty في الـ transducer).

### س145. الـ reproducibility وإدارة التجارب في تدريب الكلام: الـ seeds، الـ configs، إصدارات الـ data، الـ tracking، الـ compute budgeting، والتوثيق اللي يسمح لشخص تاني يكمّل شغلك.

**الإجابة النموذجية:**
كل run لازم يكون قابل لإعادة الإنتاج من: commit hash للكود، config كامل (بما فيه الـ augmentation والـ tokenizer والـ normalizer)، version للـ data manifests (hash)، الـ seed، وبيئة التشغيل (container image). التذبذب بين الـ seeds في الـ STT حقيقي (0.2-0.5% WER)، فالمقارنات المهمة بتتعمل بأكتر من seed أو بـ bootstrap، وأي ادعاء تحسن أقل من التذبذب ده مش ادعاء. الـ tracking (W&B أو MLflow أو حتى جدول منضبط) بيسجل الـ metrics والـ configs والـ artifacts (checkpoints مع hashes)، مع naming convention واضحة، وdashboard واحد يقارن الـ runs على نفس الـ dev. الـ compute budgeting: كل تجربة لها تقدير مسبق بالـ GPU-hours وسقف، والـ runs الكبيرة محتاجة موافقة ومراجعة config من شخص تاني قبل البدء (نص الأخطاء المكلفة بتكون config غلط في الـ data path أو الـ LR).

التوثيق: لكل مشروع ملف "run book" فيه: الهدف، الـ baseline، القرارات وأسبابها، النتائج بالأرقام على نفس الـ test، الـ runs الفاشلة وليه (دي أهم من الناجحة)، وخطوات إعادة الإنتاج. والـ checkpoints النهائية مع بطاقة موديل (model card) فيها الـ data المستخدمة (بالتراخيص)، الـ metrics بالـ slices، الحدود المعروفة، وتاريخ وإصدار.

المرشح القوي بيذكر إن أكبر مشكلة عملية هي "الـ config drift": نسخ config قديم مع تعديلات غير موثقة، والحل configs hierarchical (base + overrides) مع diff مطبوع في بداية كل run، وإن الـ resume من checkpoint لازم يتحقق إن الـ config الحالي متطابق مع اللي اتحفظ.

**سؤال متابعة:** مهندس مشى وساب 40 checkpoint من غير توثيق. ازاي تعيد بناء المعرفة بأقل تكلفة؟ (تقييم الكل على dev موحد، وقراءة الـ configs المحفوظة جوه الـ checkpoints، وتوثيق النتائج).

---

## 11. أسئلة نظرية وبحثية

الأسئلة دي بتختبر الفهم العميق للأسباب مش الاستخدام: ليه الطرق بتشتغل وليه بتفشل. مناسبة لمرشحي الـ foundation models والـ research engineers، ومش مطلوب يجاوب عليها مرشح الـ production stack.

### س146. الـ positional encodings في موديلات الكلام: absolute مقابل relative (Transformer-XL style في الـ Conformer) مقابل RoPE مقابل ALiBi. ليه الـ relative كسب في الكلام، وايه علاقة ده بالـ length extrapolation والـ streaming؟

**الإجابة النموذجية:**
الـ absolute sinusoidal (Transformer الأصلي) بيضيف vector يعتمد على الموضع المطلق، والموديل بيتعلم يعمم على الأطوال اللي شافها بس؛ في الكلام الأطوال متغيرة جدًا (من ثانية لدقايق) والمعلومة المهمة هي المسافة بين الـ frames (الـ phoneme ده جنب اللي قبله) مش الموضع المطلق، فالـ relative positional encoding (اللي بيدخل term يعتمد على i - j في حساب الـ attention scores، زي Transformer-XL اللي الـ Conformer استخدمه) بيدي invariance للإزاحة وبيعمم أحسن على الأطوال الأطول. RoPE بيعمل rotation للـ queries والـ keys بزاوية تعتمد على الموضع بحيث الـ dot product يعتمد على الفرق بس، وهو أرخص من relative attention الكامل وبقى الـ default في الـ LLMs وفي موديلات كلام حديثة (Zipformer بيستخدم نسخته الخاصة، وموديلات TTS مبنية على LLMs بتورثه). ALiBi بيضيف penalty خطي مع المسافة بدون parameters، وبيعمم على الأطوال بشكل ممتاز، بس بيفترض إن البعيد أقل أهمية دايمًا، وده منطقي جزئيًا في الكلام.

الـ length extrapolation: موديل متدرب على utterances لحد 20 ثانية بيتدهور على 60 ثانية لو الـ encoding مش بيعمم، وده سبب إضافي لتفضيل الـ relative/RoPE مع chunking أو limited context في الـ inference. في الـ streaming، الـ relative encodings بتسمح إن الـ chunk الجديد يتعالج بنفس الـ weights بغض النظر عن موضعه في المكالمة (مفيش "موضع رقم 100 ألف")، وده شرط أساسي لمكالمات طويلة، بينما الـ absolute بيحتاج تصفير أو تقطيع.

**سؤال متابعة:** ليه الـ Conformer الأصلي حط الـ positional encoding في الـ attention مش في الـ input embedding؟ وايه اللي بيحصل لو شلت الـ positional encoding خالص من موديل فيه convolution module؟ (الـ conv بيدي معلومات موضعية نسبية ضمنية، فالتدهور أقل من المتوقع).

### س147. الـ CTC بيفترض conditional independence بين الـ outputs. اشرح الفرضية بالظبط، وايه اللي بيترتب عليها (الـ peaky behavior، الحاجة لـ LM، صعوبة الـ code-switching)، وازاي الـ intermediate CTC والـ self-conditioning بيكسروها جزئيًا.

**الإجابة النموذجية:**
الـ CTC بيعرّف احتمال النص كمجموع احتمالات كل الـ alignments (مسارات من الـ tokens والـ blanks بطول T بتتقلص للنص)، واحتمال كل alignment هو حاصل ضرب احتمالات الـ frames: p(path) = Π_t p(π_t | x)، يعني كل frame بيتوقع مستقل عن الـ frames الأخرى بشرط الـ encoder output. الفرضية إن الـ label في frame t مش بيعتمد على الـ labels اللي قبله إلا من خلال الـ encoder (اللي شاف الصوت كله لو non-streaming). النتائج: (1) الموديل ما يقدرش يعبّر عن "لو الكلمة السابقة كانت X فالحالية غالبًا Y" إلا لو الـ encoder شفّرها في الـ features، فالمعرفة اللغوية محدودة وبيحتاج external LM للـ decoding، والـ LM fusion بيدي مكاسب أكبر مع CTC من الـ AED. (2) الـ peaky behavior: الموديل بيتعلم يطلّع الـ token في frame واحد وblank في الباقي، لأن ده الطريقة الأسهل لتعظيم الـ likelihood تحت الاستقلال، وده بيخلي الـ timestamps تقريبية والـ confidence غريبة. (3) التكرار اللغوي والـ code-switching صعبين لأن قرار "أكتب الكلمة دي بحروف لاتينية" محتاج اتساق عبر الـ frames.

الـ intermediate CTC بيضيف loss على layers وسطى فبيجبر التمثيلات المبكرة تكون قابلة للفك، وده regularization أساسًا. الـ self-conditioned CTC بيرجّع توزيع الـ layer الوسطى (أو الـ embedding بتاع توقعها) للـ layers اللي بعدها كمدخل إضافي، فالـ layers العليا بتشوف "مسودة" للنص وبالتالي بتقدر تعمل conditioning على الـ labels المتوقعة للأجزاء التانية، وده بيكسر الاستقلال ضمنيًا (لأن الاستقلال بشرط الـ encoder output، والـ encoder بقى شايف الـ labels). النتيجة إن CTC-only model بيقرب من الـ transducer في الدقة مع الحفاظ على الـ non-autoregressive decoding السريع.

**سؤال متابعة:** ليه الـ CTC ما بيقدرش يطلّع نفس الـ token مرتين متتاليتين من غير blank، وايه اللي ده بيفرضه على أقل عدد frames لكلمة زي "الله"؟

### س148. الـ prediction network في الـ transducer بيشتغل كـ internal language model. اشرح ليه، وليه الـ shallow fusion مع LM خارجي بيحتاج ILME/HAT، وايه علاقة الـ exposure bias بالـ AED والـ teacher forcing.

**الإجابة النموذجية:**
في الـ transducer، الـ prediction network بياخد الـ tokens السابقة فقط (من غير صوت) وبيطلّع تمثيل بيتجمع مع الـ encoder في الـ joiner. لو "شلت" الـ encoder (صفّرت مدخله أو خدت متوسطه)، الـ joiner + prediction network بيطلّعوا توزيع على الـ token التالي بناءً على النص فقط: ده هو الـ internal LM اللي الموديل اتعلمه من نصوص الـ training data. لما تعمل shallow fusion (تجمع log p_model مع λ log p_extLM)، أنت فعليًا بتضرب الـ internal LM في الـ external LM، وده double counting للـ prior اللغوي بتاع الـ training domain؛ لو الـ domain الجديد مختلف، الـ internal LM بيقاوم. الـ ILME بيقدّر الـ internal LM (بالطريقة اللي فوق) وبيطرحه بوزن: score = log p_model + λ log p_ext - μ log p_ILM، فبيسمح للـ external LM يفرض الـ domain الجديد. الـ HAT بيصمم الـ joiner بحيث احتمال الـ blank منفصل عن توزيع الـ labels، فتقدير الـ internal LM بيبقى مضبوط رياضيًا مش تقريب.

الـ exposure bias في الـ AED: في التدريب بـ teacher forcing، الـ decoder بيشوف الـ tokens الصحيحة السابقة دايمًا، وفي الـ inference بيشوف توقعاته هو، فأول غلط بيدخّله في distribution ما شافهاش في التدريب وبيتراكم (وده جزء من الـ hallucination والتكرار). العلاجات: scheduled sampling (تدريب على توقعاته أحيانًا)، minimum WER training (sequence-level loss على N-best)، والـ joint CTC decoding اللي بيرجّعه للصوت. الـ transducer أقل عرضة لأن الـ alignment مقيّد بالـ frames، والـ CTC مش عنده الظاهرة أصلًا لأنه non-autoregressive.

**سؤال متابعة:** الـ stateless prediction network (بيشوف آخر token واحد أو اتنين بس) بيقلل الـ internal LM. ايه اللي بتخسره وايه اللي بتكسبه؟

### س149. تعقيد الـ attention مع الصوت الطويل: FlashAttention، الـ local/chunked attention، الـ linear attention والـ SSMs (Mamba وأمثاله) في الكلام. ايه اللي مهم فعلًا في الـ practice؟

**الإجابة النموذجية:**
الـ self-attention تكلفته تربيعية في طول الـ sequence، وفي الكلام الـ sequence طويلة (دقيقة = 750 frame عند 80 ms، و1500 عند 40 ms، و6000 عند 10 ms بدون subsampling). أول حاجة عملية: الـ subsampling نفسه (4x ثم 8x) قلل المشكلة بأربع لـ ستة عشر مرة، وده ليه FastConformer وWhisper بيعملوا كده. FlashAttention بيحسب نفس الـ attention بالظبط بدون تخزين الـ N×N matrix في الـ HBM (tiling وrecomputation)، فبيوفر memory وبيسرّع 2-4x، وهو الآن default ولازم يكون مفعّل؛ ما بيغيرش التعقيد النظري لكن بيخلي عشرات الثواني مش مشكلة. الـ local/chunked attention (كل frame يشوف نافذة محدودة أو chunk + left context) بيخلي التكلفة خطية وبيسمح بالـ streaming والـ long-form، وبيخسر شوية دقة على الـ offline، وده اللي بتعتمد عليه الموديلات الـ streaming.

الـ linear attention والـ SSMs (Mamba وS4 وأمثالهم) بيعالجوا الـ sequence بتكلفة خطية وبـ state ثابت الحجم، وده جذاب جدًا للـ streaming (الـ state هو الـ cache بالظبط) ولأجهزة الـ edge. في الكلام فيه نتائج معقولة (موديلات ASR وTTS مبنية على Mamba وSSMs من شركات زي Cartesia للـ TTS بـ latency واطية)، لكن على الـ benchmarks الكبيرة الـ Conformer/Transformer لسه الأقوى غالبًا، والـ ecosystem (export، kernels، recipes) أنضج. الحكم العملي: للـ STT والـ TTS الحالية، subsampling + FlashAttention + chunked attention بيحلوا 95% من المشكلة، والـ SSMs تستاهل التجربة في الـ streaming TTS والأجهزة، مع التأكد من الـ export path قبل الاستثمار.

**سؤال متابعة:** ليه الـ attention التربيعي مشكلة أكبر في الـ speech LLMs من الـ STT العادي؟ (الـ audio tokens بتتضاف لـ context نصي طويل وبيتراكم عبر الـ turns، مع decoder autoregressive).

### س150. اختيارات الـ normalization والـ activation في موديلات الكلام: BatchNorm مقابل LayerNorm في الـ conv module، pre-LN مقابل post-LN، Swish/GELU مقابل snake، والـ BiasNorm في Zipformer. ليه بتفرق؟

**الإجابة النموذجية:**
الـ BatchNorm بيعتمد على إحصائيات الـ batch، وفي الكلام الـ batches فيها padding وأطوال متغيرة، فالإحصائيات ملوثة بالـ padding لو ما اتعملش masking، وفي الـ streaming والـ batch size 1 وقت الـ inference الإحصائيات المحفوظة (running stats) بتختلف عن التدريب. عشان كده الـ recipes الحديثة بتستبدل BatchNorm في الـ conv module بـ LayerNorm أو GroupNorm، خصوصًا للـ streaming. الـ pre-LN (normalization قبل الـ sublayer مع residual نظيف) أسهل في التدريب وبيتحمل LR أعلى ومش محتاج warmup طويل، والـ post-LN بيدي أحيانًا أداء أفضل شوية بس غير مستقر للموديلات العميقة؛ الـ Conformer بيستخدم pre-LN مع final LayerNorm.

الـ activations: Swish (SiLU) في الـ Conformer وGELU في Transformers شبه متكافئين، والفرق بيبقى في التفاصيل. الـ snake activation (في BigVGAN) مختلف نوعيًا: بيضيف مكوّن دوري (sin²) للـ activation، فالشبكة بتقدر تمثّل إشارات دورية (الـ harmonics) بسهولة وبيعمم أحسن على الـ pitch خارج نطاق التدريب، وده مهم للـ vocoders تحديدًا مش للـ encoders. Zipformer قدّم BiasNorm (تطبيع أبسط من LayerNorm بيحتفظ بمعلومة الـ scale اللي الـ LayerNorm بيرميها)، وactivations زي SwooshR/SwooshL مصممة عشان تتجنب مشاكل الـ Swish مع الـ negative inputs، مع تقنيات balancing لمنع الـ activations من الانفجار أو الموت، وده اللي خلاه يتدرب بـ ScaledAdam بشكل مستقر. الرسالة: الاختيارات دي مش تجميل، هي اللي بتفرق بين run بيتقارب ومش بيتقارب على الأحجام الكبيرة، بس ما بتغيرش الأداء النهائي كتير لو التدريب استقر.

**سؤال متابعة:** ليه الـ LayerNorm بيرمي معلومة الـ magnitude، وامتى دي بتهم في الكلام؟ (الـ energy معلومة مهمة للـ VAD والـ prosody).

### س151. الـ knowledge distillation نظريًا وعمليًا في الكلام: KL على التوزيعات مقابل sequence-level KD، الـ temperature، الـ layer-to-layer distillation، وليه تقطير الـ encoder أصعب من الـ decoder؟ وايه علاقة الـ quantization-aware training؟

**الإجابة النموذجية:**
الـ KD الكلاسيكي بيدرب الطالب يطابق توزيع المعلم (KL بين الـ softmax outputs بـ temperature بتنعّم التوزيع فتظهر "dark knowledge": المعلم بيقول إن الـ token الغلط ده محتمل 5%، وده معلومة أغنى من الـ one-hot). في الـ AED والـ transducer، الـ token-level KD بيحتاج نفس الـ tokenizer ونفس الـ alignment (سهل في الـ AED مع teacher forcing، أصعب في الـ transducer لأن الـ lattice مختلفة)، وفي الـ CTC الـ frame-level KD بيتأثر بالـ peaky behavior (المعلم بيطلّع blanks في مواضع مختلفة عن الطالب). الـ sequence-level KD (الطالب بيتدرب على الـ hypotheses النهائية للمعلم كـ pseudo-labels) أبسط وبيشتغل مع أي معماريتين، وهو الأكثر استخدامًا عمليًا (Distil-Whisper جمع الاتنين).

الـ layer-to-layer (مطابقة hidden states للطالب مع layers مختارة من المعلم بـ MSE أو cosine) مفيد لتقطير الـ SSL encoders (DistilHuBERT وأمثاله) وبيحتاج projection لو الأبعاد مختلفة. تقطير الـ encoder أصعب لأن الـ encoder هو اللي بيحمل الـ acoustic knowledge (robustness للضوضاء واللهجات)، وتقليل عمقه بيضرب الحالات الصعبة أولًا، بينما الـ decoder بيحمل معرفة لغوية يمكن تعويضها بـ LM أو بـ decoder صغير مع encoder قوي؛ وعشان كده Distil-Whisper وturbo حافظوا على الـ encoder كامل.

الـ quantization-aware training: بدل الـ post-training quantization اللي بيعمل mismatch بين التدريب والـ inference، بتحاكي الـ quantization (fake quant) أثناء التدريب/fine-tuning فالـ weights بتتكيف، وبيبقى مهم للـ int8/int4 على الأجهزة وللـ vocoders الحساسة. والمبدأ المشترك مع الـ KD: أي ضغط لازم يتقاس على الـ slices الصعبة (لهجات، ضوضاء) مش على المتوسط.

**سؤال متابعة:** ليه الـ pseudo-labels من معلم كبير أحيانًا بتدي طالب أحسن من التدريب على الـ labels البشرية؟ (اتساق الـ normalization والأسلوب، وتقليل label noise البشري، وتغطية data أكبر).

### س152. تحليل التمثيلات في موديلات الكلام: الـ probing حسب الـ layer (معلومات صوتية مقابل phonetic مقابل speaker مقابل semantic)، الـ ABX test، الـ phone purity للـ discrete tokens. ايه اللي بيخلي discrete token "كويس"؟

**الإجابة النموذجية:**
التحليلات المنشورة على wav2vec 2.0 وHuBERT وWavLM بتبين تدرج: الـ layers الأولى بتحمل معلومات صوتية خام وهوية المتكلم، الوسطى بتحمل أعلى معلومات phonetic (وعشان كده الـ k-means للـ HuBERT بيتعمل على layer وسطى، وعشان كده الـ voice conversion بياخد content من هناك)، والعليا بتقرب من المعنى/الكلمات في الموديلات المتدربة بأهداف لغوية (وفي wav2vec 2.0 آخر layers بترجع للـ objective فتقل فائدتها). الـ probing: تدرب classifier خطي على features كل layer لمهمة (phone classification، speaker ID، emotion) وتشوف أنهي layer أعلى، وده بيرشدك لأنهي layer تستخدم لكل مهمة ولإزاي تعمل weighted sum (SUPERB style). الـ ABX test بيقيس هل تمثيلات نفس الـ phoneme في سياقات مختلفة أقرب لبعض من تمثيلات phonemes مختلفة، بدون labels كتير، وهو المقياس الكلاسيكي في الـ zero-resource challenges.

للـ discrete tokens: (1) الـ phone purity (لكل token، نسبة الـ frames اللي بتنتمي لأشهر phoneme فيه) والـ cluster purity والعكس (phone-normalized mutual information)؛ الـ token الكويس للـ LM عالي في دول. (2) الـ bitrate: كام bit في الثانية؛ الأقل أسهل للنمذجة. (3) قابلية إعادة البناء: WER بـ ASR على الصوت المعاد من الـ tokens وspeaker similarity (للـ acoustic tokens). (4) الـ robustness: نفس الكلام بضوضاء أو متكلم مختلف يطلّع نفس الـ tokens تقريبًا (مهم للـ semantic tokens ومش مطلوب للـ acoustic). (5) استقلالية عن المتكلم للـ semantic (speaker probing على الـ tokens لازم يفشل). الـ tradeoff الجوهري: الـ token اللي بيحتفظ بكل حاجة (acoustic) صعب للـ LM، واللي بيرمي كل حاجة غير الـ phonetics (semantic) بيحتاج decoder يخترع الباقي من الـ prompt.

**سؤال متابعة:** ليه الـ semantic tokens المتدربة بالـ ASR supervision (زي S3) بتكون أحسن للـ TTS من HuBERT k-means، وايه اللي بتخسره؟ (أنظف phonetically وبتتعامل مع الضوضاء، لكنها بتفقد الـ prosody أكتر).

### س153. الإحصاء وراء المقاييس: افتراضات الـ WER، ليه الـ CER أنسب للعربي أحيانًا، مشاكل الـ MOS (تأثير المقيّم وضغط المقياس)، الـ bootstrap والدلالة، وحدود BLEU في الـ speech translation.

**الإجابة النموذجية:**
الـ WER هو edit distance على مستوى الكلمة مقسوم على عدد كلمات المرجع، وبيفترض إن كل الكلمات متساوية الأهمية (غلطة في "من" تساوي غلطة في رقم الحساب)، وإن الـ tokenization إلى كلمات محددة (وفي العربي "وبالسيارة" كلمة واحدة مع 3 morphemes، فخطأ في حرف واحد = خطأ كلمة كاملة)، وممكن يزيد عن 100% مع الـ insertions. الـ CER بيخفف مشكلة الكلمات الطويلة الملتصقة وبيعطي صورة أنعم للتدهور، وبيحتاج normalization أكتر (التشكيل والهمزات)، فالممارسة الجيدة تقرير الاتنين مع WER بعد normalization موحدة، وعلى الأقل metric واحد للـ entities.

الـ MOS: تقييم على مقياس 1-5 من مقيّمين، وفيه مشاكل إحصائية معروفة: المقيّمين بيختلفوا في الصرامة (rater effect)، والمقياس بينضغط عند الجودة العالية (كل الأنظمة الحديثة بين 4.0 و4.5 والفرق أقل من الـ noise)، والترتيب بيتأثر بالأنظمة المعروضة معًا (context effect). عشان كده الـ CMOS (مقارنة زوجية مع تفضيل بدرجة) أكثر حساسية، ولازم confidence intervals تتحسب مع أخذ الـ raters والجمل كـ random effects (mixed model) مش كأن كل تقييم مستقل، والـ attention checks والـ hidden references ضرورية.

الدلالة: الفرق بين نظامين على test set محدد عينة من الفروق الممكنة؛ الـ paired bootstrap على مستوى الـ utterances بيدي confidence interval للفرق، والـ MAPSSWE test الكلاسيكي في ASR بيعمل مقارنة زوجية. وحجم العينة بيتحدد بالفرق اللي عايز تكتشفه. BLEU في الـ AST بيقيس تطابق n-grams مع مرجع واحد، وبيعاقب الترجمات الصحيحة المختلفة الصياغة، وحساس للـ tokenization العربية جدًا؛ COMET (مبني على موديلات لغوية مدربة على أحكام بشرية) أفضل ارتباطًا بالبشر، مع الحاجة الدائمة لتقييم بشري على عينة.

**سؤال متابعة:** نظام A عنده WER أقل ونظام B عنده دقة أعلى في الأرقام. ازاي تبني metric واحد يعكس أولويات المنتج؟ (weighted WER أو reporting منفصل مع قواعد قرار صريحة).

### س154. نظريًا، امتى الـ end-to-end بيكسب الـ cascade؟ الـ error propagation، الـ information bottleneck بتاع النص، والـ joint optimization، مقابل قيود الـ data والتحكم. طبّق ده على voice agents وspeech translation.

**الإجابة النموذجية:**
الحجة النظرية للـ end-to-end: (1) النص bottleneck: لما تحوّل الصوت لنص بتفقد الـ prosody والنبرة والتردد والمتكلم والتداخل، والمهمة النهائية (الترجمة، الرد، الفهم) ممكن تحتاجها. (2) الـ error propagation: الـ cascade بياخد قرار صلب (hard decision) في كل مرحلة، وغلط الـ STT ما بيتصلحش بعده، بينما الموديل الموحد بيمرر uncertainty ضمنيًا. (3) الـ joint optimization: كل مكوّن في الـ cascade بيتحسن لهدفه المحلي (WER) مش للهدف النهائي (نجاح المهمة)، والـ end-to-end بيتدرب على الهدف النهائي مباشرة.

الحجة العملية للـ cascade: (1) الـ data: النص المتاح للتدريب أكبر بآلاف المرات من الصوت المعنون بالمهمة النهائية، والـ cascade بيستفيد من كل الـ text data (LLMs وMT)، بينما الـ end-to-end محتاج أزواج (صوت، رد) نادرة. (2) التحكم والقابلية للتفسير: كل مرحلة قابلة للاختبار والتصليح والاستبدال، والـ guardrails طبيعية. (3) الـ modularity بتسمح بأفضل موديل لكل مرحلة ولكل لغة. (4) الـ soft cascade (تمرير N-best أو الـ embeddings بدل النص) بيقلل الـ error propagation جزئيًا، والـ joint training للـ cascade ممكن.

التطبيق: في الـ speech translation، الـ end-to-end بيكسب لما الـ data كافية وpair اللغات فيه data صوتية كبيرة (الإنجليزي وأمثاله)، ولسه الـ cascade أفضل للعربي اللهجي. في الـ voice agents، الـ S2S models بتكسب في الـ naturalness والـ latency والـ paralinguistics، والـ cascade بيكسب في الدقة على المهام والتحكم والامتثال، والاتجاه إن الاتنين بيتقاربوا: cascade بمكونات streaming ومشاركة معلومات، وS2S بـ text backbone وtools. المرشح القوي بيقول إن السؤال مش أيديولوجي: بتختار حسب الـ data المتاحة وتكلفة الخطأ في الـ use case.

**سؤال متابعة:** ايه "الـ hard decision" في cascade الـ voice agent غير الـ STT؟ (الـ endpointing، الـ intent classification، والـ TTS chunking كلها قرارات صلبة بتفقد معلومات).

---

## 12. أسئلة قيادية وbusiness لمستوى lead

الأسئلة دي للمرشحين اللي هيقودوا فريق أو هيتعاملوا مع الإدارة والعملاء والجهات التنظيمية. مفيش إجابة صح واحدة، اللي بتدور عليه إن المرشح يفكر بالأرقام والقيود والمخاطر، ويعرف يقول لا بأسباب، ويربط التقنية بقرار business.

### س155. Build مقابل buy للـ STT والـ TTS في السوق السعودي: ازاي تعمل التحليل، وايه الحالات اللي الـ open weights والـ on-prem بيكسبوا فيها، وايه اللي الـ API التجاري بيكسب فيه؟

**الإجابة النموذجية:**
التحليل بيبدأ من القيود قبل الجودة: (1) الـ data residency والامتثال: لو الصوت بيانات عملاء بنك أو جهة حكومية، الـ API الأجنبي غالبًا خارج النقاش أو محتاج موافقات معقدة، وده لوحده بيحسم اتجاه on-prem أو سحابة محلية معتمدة. (2) الجودة على اللهجات: الـ APIs الكبيرة ممتازة في MSA والإنجليزي ومتفاوتة جدًا في اللهجات السعودية، ولازم تقيسها بنفسك على test set بتاعك مش على الـ demo. (3) التكلفة عند الحجم: بالدقيقة الـ API بيغلى بسرعة مع مئات المكالمات المتزامنة، والـ on-prem تكلفة ثابتة في الحديد والناس. (4) التحكم: القدرة على fine-tuning، الـ hotwords، الـ voice المخصص، وثبات السلوك (الـ API بيتغير من غير ما تعرف). (5) سرعة الوصول للسوق: الـ API بيخليك تطلق pilot في أسابيع.

الـ open weights + on-prem بيكسب لما: الحجم كبير، الامتثال صارم، اللهجة أساسية في المنتج، وعندك فريق يقدر يشغّل ويحسّن الموديلات. الـ API بيكسب لما: الحجم صغير أو متذبذب، المحتوى مش حساس، اللغة الأساسية MSA أو إنجليزي، أو الفريق صغير. والحل الواقعي كتير من المرات هجين: open weights للـ STT (لأن اللهجة والامتثال)، وTTS مخصص مبني داخليًا للـ brand voice، مع API تجاري كـ fallback أو للغات الثانوية.

المرشح القوي بيحسب total cost of ownership لثلاث سنين (بما فيها الـ retraining والـ on-call والتحديثات) مش سعر الدقيقة، وبيذكر مخاطر الـ lock-in في الاتجاهين: الـ API ممكن يغيّر السعر أو يوقف الموديل، والـ on-prem ممكن يتخلّف عن السوق لو الفريق ما لحقش، وبيقترح abstraction layer بتخلي التبديل ممكن.

**سؤال متابعة:** الـ vendor بيعرض يعمل fine-tuning على data بتاعتنا داخل سحابتهم بسعر ممتاز. ايه الأسئلة اللي تطرحها قبل الموافقة؟

### س156. ازاي تدير bake-off عادل بين موديلات أو vendors للـ STT؟ ايه الـ protocol، وايه الحيل اللي لازم تنتبه لها؟

**الإجابة النموذجية:**
الـ protocol: (1) test set داخلي مقفول ما اتشافش من أي طرف، ممثل للـ production بالـ slices (لهجات، قنوات، ضوضاء)، بـ transcripts بجودة عالية وnormalization موثقة. (2) نفس الشروط للكل: نفس الصوت بنفس الصيغة، نفس إعدادات الـ streaming (chunk size، endpointing) لو المقارنة real-time، ونفس الـ normalization في الـ scoring، بدون hotwords أو مع نفس القائمة للكل. (3) القياس على محاور: WER بالـ slices، entity accuracy، latency p50/p95، throughput أو السعر لكل ساعة، الـ hallucination rate، والـ robustness. (4) التقييم النوعي: مراجعة بشرية لعينة من أسوأ الأخطاء لكل نظام. (5) توثيق كل حاجة بحيث تتعاد بعد 6 شهور لما الموديلات تتحدث.

الحيل اللي تنتبه لها: الـ vendor بيطلب عينة من الـ data "لضبط النظام" وبعدين بيقيّم عليها أو على حاجة شبهها؛ الـ demos بجمل مختارة؛ أرقام WER منشورة على benchmarks مش ممثلة؛ الـ normalization المتساهلة (شيل التشكيل والهمزات بيقلل WER جدًا) وكل vendor بيقيس بطريقته؛ الـ latency مقاسة بدون شبكة؛ وموديل بيشتغل ممتاز على utterances قصيرة وبينهار في المكالمات الطويلة. والحيلة الداخلية: الفريق بيفضّل موديله لأنه بناه، فالـ scoring لازم يكون آلي وأعمى.

القرار في الآخر مش "الأقل WER" بل الأنسب للقيود (الامتثال، التكلفة، التحكم)، بشرط إن الفرق في الدقة مش جوهري في الـ slices الحرجة، والمرشح القوي بيكتب توصية مكتوبة بالأرقام والمخاطر مش بالانطباع.

**سؤال متابعة:** موديلين متقاربين في الـ WER الإجمالي، واحد أحسن في النجدي والتاني أحسن في الحجازي. ازاي تقرر؟

### س157. ازاي تقيس الـ ROI لـ voice agent بصدق؟ ايه الـ metrics، وازاي تصمم pilot بـ control group، وايه الفخاخ في حساب "التوفير"؟

**الإجابة النموذجية:**
الـ metrics على مستويات: (1) التشغيل: containment rate (المكالمات اللي اتحلت بدون موظف)، task completion rate لكل نوع طلب، متوسط مدة المعالجة قبل التحويل، نسبة التحويل الفوري. (2) العميل: CSAT/NPS بعد المكالمة، نسبة إعادة الاتصال خلال 24-48 ساعة لنفس الموضوع (أهم مؤشر على "الحل الوهمي")، نسبة الشكاوى. (3) المال: تكلفة المكالمة المحلولة بالـ agent مقابل الموظف، مع إضافة تكلفة المكالمات اللي اتحوّلت بعد وقت ضايع (التحويل بعد 3 دقايق أغلى من موظف من الأول). (4) الجودة والامتثال: أخطاء في المعلومات، مخالفات للسياسة.

الـ pilot: تقسيم عشوائي للمكالمات (أو الفروع أو الأوقات) بين الـ agent والمسار التقليدي، بنفس الفترة الزمنية، ومقارنة كل الـ metrics بما فيها الـ downstream (هل العميل رجع اتصل؟ هل الطلب اتنفذ فعلًا في النظام؟)، ومدة كافية عشان تشوف الذروة والحالات النادرة.

الفخاخ: (1) حساب التوفير على أساس المكالمات "المحتواة" بدون خصم اللي رجعت تاني أو اتحلت غلط. (2) الـ agent بيغري الناس تقفل وتتصل تاني أو تروح الفرع، فالتكلفة اتنقلت مش اختفت. (3) مقارنة الـ agent بأسوأ موظف. (4) تجاهل تكلفة الفريق والبنية والتحديثات. (5) الحكم من أول أسبوعين (الفضول بيرفع الاستخدام ثم بينزل). (6) الـ containment اللي ارتفع لأن الـ agent بقى بيمنع الوصول للموظف (dark pattern) وده بيرفع الشكاوى بعدين.

والمرشح على مستوى lead بيقول إن الـ ROI الأكبر أحيانًا مش في الأتمتة الكاملة لكن في الـ agent-assist والتلخيص والـ QA، وإن قرار "أتمتة ايه" لازم يبدأ من تحليل المكالمات الحقيقية مش من الرغبة في التقنية.

**سؤال متابعة:** الـ containment 65% والـ CSAT نزل 10 نقاط. ايه القرار؟

### س158. ضع roadmap لـ 12 شهر لفريق speech في شركة بتبني منتجات AI عربية: التسلسل بين الـ data والـ STT والـ TTS والـ agent والبحث، والـ milestones، وايه اللي تبنيه كـ platform وايه كـ product.

**الإجابة النموذجية:**
المبدأ: الـ data والـ evaluation قبل الموديلات، والمنتج الأول بيغذّي الـ data للي بعده. تسلسل معقول: (الشهور 1-3) بناء الـ evaluation assets: test sets لهجية للـ STT وevaluation set للـ TTS، وpipeline annotation شغّال مع vendors واثنين، وbaseline بموديلات مفتوحة على البنية الداخلية، وpilot سريع بـ agent-assist أو transcription لعميل واحد لأنه أقل مخاطر وبيجيب data حقيقية. (الشهور 3-6) الـ STT الداخلي: fine-tuning ثم continued pretraining على اللهجات، streaming serving على Triton، وأول voice agent محدود النطاق (use case واحد) على cascade. (الشهور 6-9) الـ TTS: brand voice مسجل ومدرب، lexicon وnormalization ناضجين، وتوسيع الـ agent لـ use cases أكتر مع evaluation harness كامل. (الشهور 9-12) التحسين والبحث: speech LLM للتحليلات، تجارب S2S، وتوسيع اللغات (إنجليزي بلكنات، أردو).

الـ platform مقابل الـ product: الـ platform هي الحاجات اللي بتتشارك بين كل العملاء (STT/TTS services، الـ evaluation، الـ data pipeline، الـ orchestration، الـ observability) وبتتبنى مرة واحدة بمعايير واضحة، والـ product هو الـ agent لعميل معين (prompt، tools، lexicon، integrations). الخطأ الشائع بناء platform مثالية سنة كاملة من غير عميل، أو العكس: agent لكل عميل من الصفر. القاعدة: أول عميلين بيبنوا الـ platform "بالغلط"، والتالت بيستهلكها.

الـ milestones لازم تكون قابلة للقياس (WER أقل من X على test set Y، latency p95 أقل من Z على 200 مكالمة متزامنة، agent بـ containment W في pilot)، ولازم يكون فيه هامش للـ unknowns (30% من الوقت)، ويكون واضح ايه اللي هيتوقف لو حاجة اتأخرت.

**سؤال متابعة:** الإدارة عايزة demo لـ S2S model في الشهر الأول. ازاي توفّق بين ده وبين الـ roadmap؟

### س159. بناء فريق speech: الأدوار اللي تحتاجها بالترتيب، ازاي تقيّم المرشحين فعلًا (مش بالـ CV)، وازاي تخلّي الفريق يتعلم مجال بيتغير كل 6 شهور؟

**الإجابة النموذجية:**
الأدوار بالترتيب اللي بيفرق: (1) مهندس STT/speech قوي عنده خبرة تدريب حقيقية على data كبيرة (مش fine-tuning بس) لأنه أساس كل حاجة. (2) شخص للـ data operations: يدير الـ annotation والـ guidelines والجودة، ويفضّل يكون عنده حس لغوي عربي، وده الدور الأكتر استهانة به والأكتر تأثيرًا. (3) مهندس infra/serving للـ GPUs والـ streaming والـ Kubernetes. (4) مهندس TTS (ممكن يكون نفس شخص الـ STT في البداية). (5) مهندس agents/orchestration مع conversation designer (شخص لغوي/UX مش مهندس). (6) باحث للـ speech LLMs بعدين.

التقييم: الـ CV بيقول "Whisper وNeMo"، فبتسأل عن أرقام ومشاكل: ايه الـ WER قبل وبعد وعلى ايه، ايه اللي كسر في الـ production، ايه القرار اللي ندم عليه. تمرين عملي قصير (تشخيص تسجيلات بأخطاء، أو مراجعة evaluation report فيه فخاخ) بيكشف أكتر من الأسئلة النظرية. وللمرشحين بخلفية أكاديمية بتسأل عن الـ engineering (data pipelines، serving)، وللمرشحين من الـ industry بتسأل عن الفهم (ليه الـ CTC بيتصرف كده). والمهم القدرة على قول "مش عارف" وسؤال أسئلة توضيحية.

التعلم المستمر: reading group أسبوعي على ورقة أو release، وإعادة إنتاج نتيجة واحدة مهمة كل شهرين على الـ data الداخلي، ووقت محمي للتجارب، وتدوير الناس بين STT وTTS والـ agents عشان ما يبقاش فيه single point of failure. والفريق الصغير أحسن ما يجري ورا كل موديل جديد؛ بيقيّم الجديد على الـ test sets الداخلية في يوم، ولو مش أحسن بوضوح يكمّل في خطته.

**سؤال متابعة:** مرشح قوي جدًا تقنيًا بس مش عارف يشرح فكرته لغير التقنيين. تاخده؟ ولأي دور؟

### س160. التواصل مع الإدارة العليا والجهات التنظيمية عن أنظمة الصوت: ازاي تشرح الـ WER والمخاطر بصدق من غير ما تقتل المشروع، وازاي تدير الـ demos، وترد على "ليه ما نستخدمش المساعد الصوتي بتاع الشركة الأمريكية الكبيرة؟"

**الإجابة النموذجية:**
الإدارة مش محتاجة تفهم الـ WER، محتاجة تفهم الأثر: "من كل 100 رقم حساب بيتقال بالصوت، النظام بيفهم 93 صح من أول مرة، والباقي بيتأكد منهم بالتكرار أو بلوحة الأرقام، ومفيش رقم بيتنفذ من غير تأكيد". يعني ترجمة المقاييس التقنية لمخاطر وتجربة عميل وتكلفة، مع سيناريوهات فشل محددة وايه اللي بيحصل فيها. الصدق بيبني ثقة على المدى الطويل: تقول إيه اللي الـ agent هيقدر عليه في الشهر الأول وايه اللي لأ، بدل وعد كبير وتراجع.

الـ demos: مكالمة حية بمشارك من الإدارة أخطر من فيديو مسجل لكن أصدق؛ التحضير بيكون بتضييق الـ use case وتحضير الـ fallback والاعتراف مسبقًا بحدود النظام ("لو قلت له رقم بسرعة هيطلب منك تعيده")، وتجنب الحوار المفتوح في الـ demo الأول. والـ demo دايمًا على القناة الحقيقية (تليفون) لأن اللابتوب بيخدع الطرفين.

الرد على "ليه ما نستخدمش X الأجنبي": بالأرقام على test set محلي (اللهجة)، وبالامتثال (فين الـ data بتروح)، وبالتحكم (هل نقدر نضيف صوت الشركة ونثبّت السلوك)، وبالتكلفة عند الحجم، وأحيانًا الرد الصادق إنه بالفعل أحسن في حاجات معينة وهنستخدمه فيها. الجهات التنظيمية بتهتم بالـ consent، والتسجيل، والتحقق من الهوية، والتحيز ضد فئات (كبار السن، لهجات)، وقابلية التدقيق (logs وتفسير القرارات)، والقدرة على الوصول لإنسان، فبتجهّز وثائق لكل نقطة قبل ما تتسأل.

**سؤال متابعة:** المدير التنفيذي شاف demo لموديل S2S عالمي وبيسأل ليه بتاعنا أقل طبيعية. ايه الرد اللي يجمع الصدق والاتجاه؟

### س161. امتى توقف أو تغيّر اتجاه مشروع صوتي؟ التعامل مع الـ hype (S2S، "الـ agent هيحل كل المكالمات")، إدارة التوقعات بعد demo مبهر، وايه الأصل الدفاعي الحقيقي (الـ moat) لشركة بتبني voice AI عربي؟

**الإجابة النموذجية:**
معايير التوقف أو التغيير لازم تتحدد قبل البداية: لو الـ pilot ما وصلش لـ task completion X أو الـ CSAT نزل عن Y بعد شهرين من التحسين، نوقف أو نضيّق النطاق. العلامات اللي بتقول غيّر الاتجاه: التحسينات بقت تدريجية جدًا على مقياس لسه بعيد عن الهدف، المستخدمين بيتحايلوا على النظام للوصول لموظف، أو الحل الأبسط (IVR محسّن، agent-assist) بيدي 80% من القيمة بـ 20% من المخاطر. والصعوبة الحقيقية إنسانية: الفريق مستثمر عاطفيًا، فالقرار لازم يتاخد بالأرقام المتفق عليها مسبقًا.

الـ hype: كل demo لـ S2S مبهر لأنه بيقاس على المحادثة الحرة، والـ production بيتقاس على المهام الحرجة. الرد العملي: تجرب الجديد بسرعة على الـ evaluation بتاعك (يوم أو اتنين)، وتكتب النتيجة، وتقرر بناءً عليها، بدل الرفض أو الاندفاع. والتوقعات بتتدار بجداول واضحة: "الطبيعية" شيء و"الدقة في المهمة" شيء، وبتعرض الاتنين.

الـ moat: مش الموديل (بيتقدم كل شهر وأي حد يقدر ينزّله)، لكن: (1) الـ data واللهجات: آلاف الساعات معنونة بجودة على اللهجات المستهدفة، وevaluation sets ما حدش عنده زيها، وvoices مسجلة بحقوق واضحة. (2) الـ integration والـ workflows: الربط مع أنظمة العملاء والـ tools وقواعد الـ business اللي بتاخد شهور. (3) الثقة والامتثال: التصاريح والسجل مع الجهات التنظيمية والعملاء الكبار. (4) المعرفة التشغيلية: الفريق اللي يعرف ليه المكالمات بتفشل. والمرشح القوي بيقول إن الأصل ده بيتبني بالتراكم مع كل عميل، وإن السرعة في التعلم من الـ production أهم من أي معمارية.

**سؤال متابعة:** شركة عالمية أطلقت STT عربي بلهجات أحسن من بتاعنا بفارق واضح. ايه اللي يتغير في الاستراتيجية وايه اللي يفضل؟

---

## 13. أسئلة سريعة للـ screening

الأسئلة دي للـ screening السريع (تليفون 20 دقيقة) أو لتسخين المقابلة. الإجابة المتوقعة جملة أو اتنين، والهدف إنك تعرف بسرعة لو المرشح شغّال فعلًا في المجال ولا قرأ عنه بس. لو غلط في 3 أو 4 من 15 سؤال عشوائي من هنا، غالبًا مش senior في الـ speech.

1. **ايه الـ RTF؟** نسبة وقت المعالجة لمدة الصوت. RTF 0.1 يعني دقيقة صوت بتتعالج في 6 ثواني. في الـ streaming بيهمنا الـ latency كمان مش الـ RTF بس.

2. **ايه الـ DER وايه دور الـ collar؟** Diarization Error Rate = (missed speech + false alarm + speaker confusion) / إجمالي وقت الكلام. الـ collar (غالبًا 250 ms حوالين حدود الـ segments) بيستثني المناطق اللي الـ annotators نفسهم مش متفقين عليها.

3. **ايه دور الـ blank في الـ CTC؟** بيسمح للموديل يطلّع "مفيش token جديد" في الـ frame، وبيفصل الـ tokens المتكررة الحقيقية (زي حرفين متتاليين متطابقين) عن التكرار الناتج عن امتداد الصوت.

4. **ليه 16 kHz هو المعيار للـ STT؟** بيغطي لحد 8 kHz (Nyquist)، وده كافي لكل الـ phonetic information تقريبًا، وأي زيادة بتكلّف compute من غير فايدة للفهم.

5. **ايه الـ exposure bias؟** في التدريب بـ teacher forcing الموديل بيشوف الـ tokens الصحيحة السابقة، وفي الـ inference بيشوف توقعاته هو، فأي غلطة بتتراكم. بيظهر في الـ AED والـ AR TTS.

6. **الفرق بين MOS وCMOS؟** الـ MOS تقييم مطلق من 1 لـ 5، والـ CMOS مقارنة زوجية بين نظامين (من -3 لـ +3) وأكثر حساسية للفروق الصغيرة.

7. **ايه الـ RIR؟** Room Impulse Response: استجابة الغرفة لنبضة، والـ convolution بيها بيحاكي الـ reverberation. بتتستخدم في الـ augmentation.

8. **الفرق بين dBFS وLUFS؟** الـ dBFS مستوى الـ peak نسبة لأقصى قيمة رقمية، والـ LUFS مقياس للـ perceived loudness مع فلتر إدراكي وgating للصمت.

9. **phone مقابل phoneme مقابل grapheme؟** الـ grapheme حرف مكتوب، الـ phoneme أصغر وحدة صوتية بتفرّق المعنى في اللغة، الـ phone التحقق الصوتي الفعلي (نطق ق كـ g أو q phones لنفس الـ phoneme).

10. **ايه الـ emission latency في الـ streaming STT؟** الفرق بين وقت نطق الكلمة فعليًا ووقت ظهورها في الـ output. الـ transducer المتدرب عادي بيميل يتأخر عشان يشوف context أكتر.

11. **ايه الـ jitter buffer؟** buffer عند المستقبل بيعوّض تفاوت وصول الـ packets عشان التشغيل يبقى متصل، وحجمه tradeoff بين latency وانقطاعات.

12. **ايه الـ DTX والـ FEC في Opus؟** الـ DTX بيوقف إرسال packets في الصمت لتوفير الـ bandwidth، والـ FEC بيبعت نسخة منخفضة الجودة من الـ frame السابق عشان يعوّض الـ packet loss.

13. **ايه الـ RVQ؟** Residual Vector Quantization: كل codebook بيقنطر الـ residual اللي فضل من اللي قبله، فالـ codebook الأول بيمسك أهم المعلومات والباقي تفاصيل.

14. **ليه Whisper بيطلّع 1500 frame للـ 30 ثانية؟** الـ mel عند 100 frame/s بيمر على conv بـ stride 2، فبيبقى 50 frame/s، و30 ثانية = 1500.

15. **ايه الـ internal LM في الـ transducer؟** الـ prediction network بيتعلم توزيع النص لوحده، وده بيتعارض مع LM خارجي في الـ fusion، فبتطرح تقدير الـ ILM (ILME) أو تستخدم HAT.

16. **الفرق بين beam search وprefix beam search في الـ CTC؟** الـ prefix beam search بيجمّع كل الـ alignments اللي بتؤدي لنفس النص (بعد شيل الـ blanks والتكرار) في hypothesis واحدة، بدل ما يعاملها كمسارات مختلفة.

17. **ايه الـ word boosting؟** إضافة score للـ hypotheses اللي فيها كلمات من قائمة أثناء الـ decoding، لرفع احتمال المصطلحات النادرة بدون retraining.

18. **ايه اللي SpecAugment بيعمله؟** بيمسك مناطق من الـ mel في الزمن والتردد (وأحيانًا time warping) أثناء التدريب كـ regularization.

19. **ليه speed perturbation بيغيّر الـ pitch؟** لأنه resampling (بيمط أو يضغط الإشارة)، فبيغيّر السرعة والتردد معًا؛ الـ tempo perturbation بيغيّر السرعة بس.

20. **ايه اللي اتغير في Whisper large-v3 عن v2؟** 128 mel بدل 80، token للكانتونية، وتدريب على data أكبر بكتير منها pseudo-labeled بموديل v2.

21. **الفرق بين semantic وacoustic tokens في جملة؟** الـ semantic بتحمل المحتوى (قريبة من الـ phonemes) وبتفقد الصوت، والـ acoustic بتحمل كل التفاصيل الصوتية وبتحتاج codebooks أكتر.

22. **ايه الـ EER؟** النقطة اللي فيها false accept rate = false reject rate في نظام التحقق. مقياس واحد للمقارنة، بس نقطة التشغيل الفعلية بتتحدد من الـ use case.

23. **ايه الـ vocoder؟** الموديل اللي بيحوّل التمثيل الوسيط (mel أو codec tokens) لـ waveform، وأساسًا بيعيد بناء الـ phase.

24. **ايه الـ G2P؟** Grapheme-to-Phoneme: تحويل النص المكتوب لتسلسل phonemes. في العربي بيحتاج تشكيل أولًا لأن الحروف بدون تشكيل مش كافية.

25. **مشكلة همزة الوصل في الـ TTS؟** "ال" و"ابن" و"استخدم" همزتها بتسقط في الوصل (وانطلق) وبتتنطق في الابتداء، والـ G2P لازم يعرف السياق والوقف.

26. **الفرق بين TN وITN؟** الـ TN (للـ TTS) بيحوّل الرموز والأرقام لكلمات منطوقة، والـ ITN (للـ STT) بيعمل العكس من الكلمات للصيغة المكتوبة.

27. **ايه الـ speaker confusion في الـ DER؟** الوقت اللي فيه الكلام اتكشف صح لكن اتنسب لمتكلم غلط. بيختلف عن الـ missed speech والـ false alarm.

28. **ايه الـ hangover في الـ VAD؟** فترة بعد آخر frame فيه كلام النظام بيفضل فيها يعتبر الكلام مستمر، عشان ما يقطعش في الوقفات القصيرة جوه الكلمة.

29. **الفرق بين AEC وNS في جملة؟** الـ AEC بيشيل صوت السماعة الراجع للمايك باستخدام الإشارة المرجعية، والـ NS بيشيل الضوضاء بدون مرجع.

30. **ليه الـ CTC peaky؟** لأن الـ loss بيكافئ alignment واحد حاد مع blanks كتير حواليه، فالموديل بيطلّع الـ token في frame واحد بثقة عالية والباقي blank.

31. **ايه الـ TTFB في الـ TTS؟** time to first byte (أو first audio): الوقت من إرسال النص لأول chunk صوتي، وهو اللي بيحدد الإحساس بسرعة الرد.

32. **ليه p95 أهم من المتوسط في الـ latency؟** لأن المستخدم بيفتكر أسوأ التجارب، والمتوسط بيخبي إن 5% من الـ turns بتاخد 3 ثواني.

33. **ايه الـ warm transfer؟** تحويل المكالمة بعد التواصل مع الموظف المستقبل وتسليمه السياق، عكس الـ cold transfer اللي بيرمي المكالمة في الطابور.

34. **الفرق بين SIP وRTP؟** الـ SIP بروتوكول الإشارة (بدء وإنهاء المكالمة والتفاوض على الـ codec)، والـ RTP بينقل الصوت نفسه.

35. **ايه الـ MIG؟** تقسيم GPU واحد (A100/H100) لـ instances معزولة في الـ memory والـ compute، مناسب للـ workloads الصغيرة.

36. **ليه الـ KV cache بيحدد الـ concurrency في الـ AR TTS؟** كل stream بيحتفظ بالـ keys/values لكل token اتولّد، فالـ memory بتتزايد مع الطول وعدد الـ streams، وبتخلص قبل الـ compute.

37. **الفرق بين fp16 وbf16 في التدريب؟** الـ bf16 عنده نفس مدى الـ fp32 (exponent أكبر) بدقة أقل، فبيتجنب الـ overflow اللي بيعمل NaN في fp16، بس محتاج hardware بيدعمه.

38. **ايه الـ gradient accumulation وامتى تحتاجه؟** تجميع الـ gradients من عدة batches صغيرة قبل التحديث لمحاكاة batch كبير لما الـ memory مش كافية.

39. **ايه الـ checkpoint averaging؟** متوسط الـ weights من آخر N checkpoints، بيدي موديل أكثر استقرارًا وغالبًا WER أحسن بشكل مجاني.

40. **ايه الوقف (pausal form) وليه مهم للـ TTS العربي؟** نهاية الكلمة بتتغير عند الوقف (سقوط الحركة الإعرابية، التاء المربوطة بتتنطق هاء)، والموديل لازم يعرف الكلمة في نهاية الجملة ولا في وسطها.

41. **ايه الـ cpWER؟** WER للـ multi-speaker: بتلصق كلام كل متكلم وبتحسب الـ WER مع أحسن تطابق (permutation) بين المتكلمين المرجعيين والمكتشفين.

42. **ايه الـ test set contamination؟** وجود نفس الصوت أو النص أو المتكلم في التدريب والاختبار، فالنتيجة بتبقى متفائلة كذبًا.

43. **الفرق بين zero-shot وfew-shot cloning؟** الـ zero-shot بيستخدم ثواني من الصوت كـ prompt بدون تدريب، والـ few-shot بيعمل fine-tuning سريع على دقايق.

44. **مشكلة الحروف الشمسية في الـ G2P؟** لام التعريف بتتدغم في الحرف الشمسي (الشمس تتنطق أش-شمس) وبتتنطق مع القمرية، والقاعدة لازم تتطبق قبل توليد الـ phonemes.

45. **ايه Buckwalter transliteration؟** نظام لكتابة العربي بحروف ASCII واحد لواحد (بما فيها التشكيل)، مفيد كتمثيل داخلي للـ G2P والـ tooling القديم.

---

## 14. تمارين عملية وcoding

التمارين دي للجزء العملي (live coding 45 دقيقة أو take-home يوم واحد). المهم مش إن الكود يشتغل من أول مرة، المهم القرارات: إزاي بيتعامل مع الحالات الحدية، إيه اللي بيقيسه، وإزاي بيتحقق من الصحة. كل تمرين معاه اللي بتدور عليه وsnippet مرجعي تقدر تقارن بيه أو تديه للمرشح كنقطة بداية.

### س162. تمرين: نفّذ CTC greedy decoding وحساب WER بـ normalization عربي.

**المطلوب:** دالة بتاخد مصفوفة log-probabilities بشكل (T, V) وقائمة الـ vocab وبترجّع النص بعد شيل الـ blank والتكرار، ودالة WER بتطبّع النصين (توحيد الهمزات والتاء المربوطة وإزالة التشكيل) قبل الحساب، مع إرجاع تفصيل الـ substitutions والـ deletions والـ insertions.

**اللي بتدور عليه:** فهم إن الـ blank بيفصل التكرارات الحقيقية، إن الـ normalization لازم تتطبق على المرجع والناتج معًا، إن الـ edit distance على مستوى الكلمات مش الحروف للـ WER، والتعامل مع النص الفاضي (قسمة على صفر).

```python
import re
import numpy as np

def ctc_greedy(logprobs: np.ndarray, vocab: list[str], blank: int = 0) -> str:
    best = logprobs.argmax(axis=1)
    out, prev = [], None
    for idx in best:
        if idx != blank and idx != prev:
            out.append(vocab[idx])
        prev = idx
    return "".join(out).replace("▁", " ").strip()

DIACRITICS = re.compile(r"[\u064B-\u0652\u0670]")
def normalize_ar(text: str) -> str:
    text = DIACRITICS.sub("", text)
    text = re.sub("[إأآا]", "ا", text)
    text = text.replace("ة", "ه").replace("ى", "ي")
    return re.sub(r"\s+", " ", text).strip()

def wer(ref: str, hyp: str) -> dict:
    r, h = normalize_ar(ref).split(), normalize_ar(hyp).split()
    d = np.zeros((len(r) + 1, len(h) + 1), dtype=int)
    d[:, 0] = np.arange(len(r) + 1); d[0, :] = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i-1] == h[j-1] else 1
            d[i, j] = min(d[i-1, j] + 1, d[i, j-1] + 1, d[i-1, j-1] + cost)
    return {"wer": d[-1, -1] / max(len(r), 1), "errors": int(d[-1, -1]), "ref_words": len(r)}
```

**سؤال متابعة:** الـ normalization دي بتخلي "ه" و"ة" واحد، وده بيخبي غلط فعلي في كلمات زي "مدرسه/مدرسة". امتى تقبل ده وامتى لأ؟

### س163. تمرين: اكتب endpointing state machine على stream من احتمالات VAD.

**المطلوب:** generator بياخد احتمالات VAD لكل frame (32 ms) وبيطلّع أحداث: speech_start، speech_end (endpoint)، مع hysteresis (threshold للدخول أعلى من الخروج)، min speech duration (عشان الضوضاء القصيرة ما تعملش start)، وmin silence duration للـ endpoint، وmax utterance length (يقطع بالقوة بعد 30 ثانية).

**اللي بتدور عليه:** الـ hysteresis، إن الـ timestamps تتحسب من الـ frames مش من وقت الوصول، التعامل مع الـ speech القصير جدًا (يتلغى بدل ما يطلع start ثم end)، وإن الـ endpoint timestamp هو آخر كلام مش لحظة القرار.

```python
def endpointer(probs, frame_ms=32, on=0.6, off=0.4, min_speech_ms=200,
               min_silence_ms=600, max_utt_ms=30000):
    in_speech, speech_frames, silence_frames, start_t = False, 0, 0, None
    for i, p in enumerate(probs):
        t = i * frame_ms
        if not in_speech:
            speech_frames = speech_frames + 1 if p >= on else 0
            if speech_frames * frame_ms >= min_speech_ms:
                in_speech, start_t, silence_frames = True, t - min_speech_ms, 0
                yield ("speech_start", start_t)
        else:
            silence_frames = silence_frames + 1 if p < off else 0
            forced = t - start_t >= max_utt_ms
            if silence_frames * frame_ms >= min_silence_ms or forced:
                end_t = t - silence_frames * frame_ms if not forced else t
                yield ("speech_end", end_t, "forced" if forced else "silence")
                in_speech, speech_frames, silence_frames = False, 0, 0
```

**سؤال متابعة:** ازاي تخلي الـ min_silence_ms adaptive على أساس الـ partial transcript (جملة ناقصة = استنى أكتر)؟

### س164. تمرين: pipeline بسيط لـ streaming transcription بالـ VAD gating.

**المطلوب:** برنامج بيقرا صوت من ملف أو مايك بـ chunks 100 ms، بيشغّل VAD، بيجمّع الكلام في buffer، ولما يحصل endpoint بيبعت الـ buffer لموديل STT (faster-whisper أو أي موديل متاح) ويطبع النتيجة مع timestamps، مع partial results كل ثانية على الـ buffer الحالي (اختياري). لازم يشتغل في thread أو async بحيث القراءة ما تتوقفش أثناء الـ inference.

**اللي بتدور عليه:** الفصل بين thread القراءة وthread الـ inference بـ queue، عدم فقدان الصوت أثناء الـ inference، تحويل الـ timestamps من نسبية للـ segment لمطلقة، حد أقصى لطول الـ buffer، وإدراك إن الـ partial على buffer كامل مكلف (وإن الـ LocalAgreement أو موديل streaming حقيقي هو الحل الصحيح).

```python
import queue, threading, numpy as np

audio_q, seg_q = queue.Queue(), queue.Queue()

def reader(chunks):                      # chunks: iterable of 1600-sample int16 arrays (100 ms @16k)
    for c in chunks: audio_q.put(c)
    audio_q.put(None)

def segmenter(vad_prob, endpointer_fn):
    buf, t0, frames = [], 0.0, []
    while (c := audio_q.get()) is not None:
        buf.append(c); frames.append(vad_prob(c))
        for ev in endpointer_fn(frames[-1:]):    # في الواقع بتمرر الـ state، ده تبسيط
            if ev[0] == "speech_end":
                seg_q.put((t0, np.concatenate(buf))); buf, frames = [], []
        t0 += 0.1 if not buf else 0
    seg_q.put(None)

def transcriber(model):
    while (item := seg_q.get()) is not None:
        start, audio = item
        for s in model.transcribe(audio.astype(np.float32) / 32768, language="ar")[0]:
            print(f"[{start + s.start:.2f} - {start + s.end:.2f}] {s.text}")
```

**سؤال متابعة:** لو الـ inference بياخد 3 ثواني والمستخدم بيتكلم باستمرار، ايه اللي هيحصل للـ queue وازاي تتعامل؟

### س165. تمرين: احسب mel spectrogram من الصفر وطابقه مع torchaudio.

**المطلوب:** دالة بتحسب log-mel بـ numpy (framing بـ window 25 ms وhop 10 ms، Hann window، FFT 512، mel filterbank 80 بين 0 و8000 Hz بمقياس HTK أو Slaney)، وتقارنها بـ torchaudio.transforms.MelSpectrogram بنفس الإعدادات، وتشرح أي فرق.

**اللي بتدور عليه:** فهم الـ padding (center=True بيضيف نص window من كل جهة)، إن الـ power spectrum بيتحسب بالتربيع، إن الـ mel scale ليها صيغتين (HTK وSlaney) والـ normalization للـ filters بتختلف، وإن log(x + eps) مقابل log10 وclamp بيفرقوا، وإن "مطابقة تقريبية" مش كافية لموديل متدرب.

```python
import numpy as np

def hz_to_mel(f): return 2595 * np.log10(1 + f / 700)          # HTK
def mel_to_hz(m): return 700 * (10 ** (m / 2595) - 1)

def mel_filterbank(sr=16000, n_fft=512, n_mels=80, fmin=0, fmax=8000):
    mels = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    bins = np.floor((n_fft + 1) * mel_to_hz(mels) / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        l, c, r = bins[m-1], bins[m], bins[m+1]
        fb[m-1, l:c] = (np.arange(l, c) - l) / max(c - l, 1)
        fb[m-1, c:r] = (r - np.arange(c, r)) / max(r - c, 1)
    return fb

def log_mel(x, sr=16000, n_fft=512, win=400, hop=160, n_mels=80):
    x = np.pad(x, (n_fft // 2, n_fft // 2), mode="reflect")            # center=True
    n_frames = 1 + (len(x) - n_fft) // hop
    window = np.hanning(win + 1)[:-1]                                     # periodic Hann
    frames = np.stack([x[i*hop:i*hop+win] * window for i in range(n_frames)])
    spec = np.abs(np.fft.rfft(frames, n=n_fft)) ** 2
    return np.log(spec @ mel_filterbank(sr, n_fft, n_mels).T + 1e-10)
```

**سؤال متابعة:** الـ torchaudio افتراضيًا بيستخدم Slaney أو HTK؟ وايه اللي يحصل لو الموديل اتدرب بواحد والـ serving بالتاني؟

### س166. تمرين: chunker للنص العربي عشان الـ streaming TTS.

**المطلوب:** دالة بتاخد stream من الـ tokens (زي اللي بييجي من LLM) وبتطلّع chunks للـ TTS: تقطع عند علامات الترقيم القوية (. ؟ !) وعند الفاصلة لو الـ chunk أطول من حد معين، بحد أدنى لطول الـ chunk (عشان ما تبعتش كلمتين لوحدهم)، ومن غير ما تقطع جوه رقم أو تاريخ أو اسم بين قوسين أو اختصار بنقطة، مع إرسال الباقي لما الـ stream يخلص.

**اللي بتدور عليه:** التفكير في الحالات الحدية (الأرقام العشرية "3.5"، "د." كاختصار، علامات الترقيم العربية "،" و"؛" و"؟")، الـ tradeoff بين chunks صغيرة (latency واطية، prosody مقطعة) وكبيرة، وإن أول chunk المفروض يطلع أسرع من الباقي.

```python
import re
STRONG, WEAK = set(".؟!?"), set("،,;؛:")
ABBREV = re.compile(r"(^|\s)(د|أ|م|ص|ش)\.$")

def chunker(token_stream, min_chars=25, soft_max=80, first_min=12):
    buf, first = "", True
    def ready():
        limit = first_min if first else min_chars
        return len(buf.strip()) >= limit and not ABBREV.search(buf) \
               and not re.search(r"\d[.,]$", buf)
    for tok in token_stream:
        buf += tok
        last = buf.rstrip()[-1:] if buf.strip() else ""
        if (last in STRONG and ready()) or (last in WEAK and len(buf) >= soft_max and ready()):
            yield buf.strip(); buf, first = "", False
    if buf.strip():
        yield buf.strip()
```

**سؤال متابعة:** الـ LLM بيطلّع "الرقم هو 0551234567." كـ tokens متفرقة "055" "123" "4567". ايه اللي ممكن يتكسر في الـ chunker ده وازاي تحميه؟

### س167. تمرين: تحليل logs لمكالمات voice agent وتحديد الـ bottleneck.

**المطلوب:** ملف JSONL فيه أحداث لكل turn: call_id، turn_id، ونوع الحدث (user_speech_end، endpoint_detected، stt_final، llm_first_token، llm_last_token، tts_first_byte، audio_playback_start) مع timestamps. اكتب سكريبت يحسب لكل مرحلة p50 وp95، ويرتّب المراحل حسب مساهمتها في الـ p95 الكلي، ويكشف الـ turns اللي فيها أحداث ناقصة أو ترتيب غلط، ويطلّع الـ latency حسب رقم الـ turn في المكالمة (هل بتزيد مع الطول؟).

**اللي بتدور عليه:** حساب الفرق بين الأحداث المتتالية مش الفرق عن البداية بس، الانتباه إن p95 الكلي مش مجموع p95 المراحل، التعامل مع الأحداث الناقصة كإشارة مشكلة مش كـ NaN يتجاهل، والقدرة على قول "المشكلة في الـ endpointing مش في الـ LLM" من الأرقام.

```python
import json, collections, numpy as np
ORDER = ["user_speech_end","endpoint_detected","stt_final","llm_first_token",
         "llm_last_token","tts_first_byte","audio_playback_start"]

turns = collections.defaultdict(dict)
for line in open("events.jsonl", encoding="utf-8"):
    e = json.loads(line); turns[(e["call_id"], e["turn_id"])][e["event"]] = e["ts"]

stages, bad, total = collections.defaultdict(list), [], []
for key, ev in turns.items():
    if any(k not in ev for k in ORDER) or any(ev[a] > ev[b] for a, b in zip(ORDER, ORDER[1:])):
        bad.append(key); continue
    for a, b in zip(ORDER, ORDER[1:]):
        stages[f"{a}->{b}"].append(ev[b] - ev[a])
    total.append(ev["audio_playback_start"] - ev["user_speech_end"])

print(f"turns={len(turns)} bad={len(bad)} total p50={np.percentile(total,50):.0f} p95={np.percentile(total,95):.0f} ms")
for name, v in sorted(stages.items(), key=lambda kv: -np.percentile(kv[1], 95)):
    print(f"{name:45s} p50={np.percentile(v,50):7.0f} p95={np.percentile(v,95):7.0f} ms")
```

**سؤال متابعة:** الـ p95 الكلي 2.1 ثانية والـ llm_first_token مساهمته الأكبر، بس الـ p50 بتاعه ممتاز. ايه اللي ده بيقوله عن طبيعة المشكلة؟ (queueing تحت الضغط أو context طويل في مكالمات معينة، مش بطء عام).

### س168. تمرين: forced alignment بالـ CTC عشان word timestamps.

**المطلوب:** باستخدام موديل CTC (مثلًا wav2vec2 أو MMS aligner من torchaudio) وtranscript معروف، طلّع بداية ونهاية كل كلمة بالثواني، وتعامل مع الكلمات اللي مش في الـ vocab (transliteration أو تقسيم لحروف)، واعرض النتيجة على شكل SRT.

**اللي بتدور عليه:** فهم إن الـ alignment بيحتاج التوكنز تطابق الـ vocab بتاع الموديل (normalization متطابقة)، إن الـ frame rate بتاع الموديل (20 ms لـ wav2vec2) بيحدد الدقة، إن الـ CTC peaky فالبداية والنهاية تقديرية، والتعامل مع التسجيل الطويل بالتقطيع.

```python
import torch, torchaudio
from torchaudio.functional import forced_align, merge_tokens

def align_words(model, labels: dict, waveform, sr, words: list[str], frame_ms=20):
    with torch.inference_mode():
        emissions, _ = model(waveform)                       # (1, T, V) logits
        logprobs = torch.log_softmax(emissions, dim=-1)
    tokens = [labels[c] for w in words for c in w.replace(" ", "|")] # حسب vocab الموديل
    targets = torch.tensor([tokens], dtype=torch.int32)
    alignment, scores = forced_align(logprobs, targets, blank=0)
    spans = merge_tokens(alignment[0], scores[0].exp())      # token spans مع scores
    out, i = [], 0
    for w in words:
        n = len(w)
        ws = spans[i:i+n]; i += n + 1                        # +1 للفاصل "|"
        out.append((w, ws[0].start * frame_ms / 1000, ws[-1].end * frame_ms / 1000))
    return out

def to_srt(items):
    fmt = lambda s: f"{int(s//3600):02d}:{int(s%3600//60):02d}:{s%60:06.3f}".replace(".", ",")
    return "\n".join(f"{k+1}\n{fmt(a)} --> {fmt(b)}\n{w}\n" for k, (w, a, b) in enumerate(items))
```

**سؤال متابعة:** الـ alignment فشل (score واطي جدًا) على جزء من التسجيل. ايه الأسباب المحتملة، وازاي تكتشف إن الـ transcript نفسه غلط في الجزء ده؟

---

## ملاحظات سريعة للتقييم

- **المستوى المتوقع من senior:** يجاوب بعمق على الـ STT أو TTS (حسب تخصصه) ويفهم الباقي على مستوى الـ tradeoffs، ويقدر يعمل system design معقول.
- **المستوى المتوقع من staff/lead:** يربط القرارات التقنية بالتكلفة والامتثال والـ data strategy، ويعرف حدود الموديلات في العربي بالتحديد، وعنده رأي مبني على تجربة مش على أوراق.
- **العلامات الإيجابية:** بيسأل أسئلة توضيحية قبل الـ design، بيذكر القياس قبل الحل، بيفرّق بين اللي جربه واللي قرأه.
- **العلامات السلبية:** كل حاجة "بتتحل بموديل أكبر"، تجاهل الـ latency والـ tail، عدم معرفة إن الـ licenses بتفرق، والتعامل مع العربي كأنه إنجليزي بحروف مختلفة.
- **الأقسام 10 إلى 14:** القسم 10 بيفرّق بين اللي درّب موديلات كبيرة فعلًا واللي عمل fine-tuning بس؛ القسم 11 بيكشف الفهم مقابل الحفظ؛ القسم 12 لمستوى lead بس ومفيش داعي تسأله لمهندس senior؛ القسم 13 للـ screening السريع مش للحكم النهائي؛ والقسم 14 بيقيس القرارات في الكود مش الإتقان النحوي.
- **التوزيع المقترح لمقابلة كاملة (3 جولات):** جولة screening بالقسم 13 زائد سؤال خبرة من القسم 9، جولة تقنية عميقة من الأقسام حسب التخصص مع system design، وجولة عملية من القسم 14 مع نقاش في الـ tradeoffs.
