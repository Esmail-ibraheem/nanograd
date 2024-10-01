LOCALES = {
    "lang": {
        "en": {
            "label": "Lang",
        },
        "ar": {
            "label": "اللغة",
        },
        
        
    },
    "model_name": {
        "en": {
            "label": "Model name",
        },
        "ar": {
            "label": "اسم المودل",
        },

    },
    "model_path": {
        "en": {
            "label": "Model path",
            "info": "Path to pretrained model or model identifier from Hugging Face.",
        },
        "ar": {
            "label": "مسار المودل",
            "info": "مسار المودل المدرب",
        },

    },
    "finetuning_type": {
        "en": {
            "label": "Finetuning method",
        },
        "ar": {
            "label": "طريقة التدريب",
        },

    },
    "checkpoint_path": {
        "en": {
            "label": "Checkpoint path",
        },
        "ar": {
            "label": "مسار ال Checkpoint",
        },

        
    },
    "advanced_tab": {
        "en": {
            "label": "Advanced configurations",
        },
        "ar": {
            "label": "اعدادت متقدمة",
        },
 
        
    },
    "quantization_bit": {
        "en": {
            "label": "Quantization bit",
            "info": "Enable quantization (QLoRA).",
        },
        "ar": {
            "label": "Quantization bit",
            "info": "Enable quantization (QLoRA).",
        },
        

        
    },
    "quantization_method": {
        "en": {
            "label": "Quantization method",
            "info": "Quantization algorithm to use.",
        },
        "ar": {
            "label": "Quantization method",
            "info": "Quantization algorithm to use.",
        },
        

        
    },
    "template": {
        "en": {
            "label": "Prompt template",
            "info": "The template used in constructing prompts.",
        },
        "ar": {
            "label": "نموذج التلقين",
            "info": "النماذج المستخدمة في انشاء التلقينات",
        },
        

    },
    "rope_scaling": {
        "en": {
            "label": "RoPE scaling",
        },
        "ar": {
            "label": "RoPE scaling",
        },

    },
    "booster": {
        "en": {
            "label": "Booster",
        },
        "ar": {
            "label": "التعزيز",
        },
 
        
    },
    "training_stage": {
        "en": {
            "label": "Stage",
            "info": "The stage to perform in training.",
        },
        "ar": {
            "label": "مرحلة",
            "info": "مرحلة الأداء في التدريب",
        },
        
  
    },
    "dataset_dir": {
        "en": {
            "label": "Data dir",
            "info": "Path to the data directory.",
        },
        "ar": {
            "label": "مسار البيانات",
            "info": "مسار البيانات",
        },
        

        
    },
    "dataset": {
        "en": {
            "label": "Dataset",
        },
        "ar": {
            "label": "البيانات",
        },

        
    },
    "data_preview_btn": {
        "en": {
            "value": "Preview dataset",
        },
        "ar": {
            "value": "معاينة البيانات",
        },
  
        
    },
    "preview_count": {
        "en": {
            "label": "Count",
        },
        "ar": {
            "label": "العدد",
        },
        

        
    },
    "page_index": {
        "en": {
            "label": "Page",
        },
        "ar": {
            "label": "الصفحة",
        },

        
    },
    "prev_btn": {
        "en": {
            "value": "Prev",
        },
        "ar": {
            "value": "السابق",
        },

        
    },
    "next_btn": {
        "en": {
            "value": "Next",
        },
        "ar": {
            "value": "التالي",
        },

    },
    "close_btn": {
        "en": {
            "value": "Close",
        },
        "ar": {
            "value": "اغلاق",
        },

    },
    "preview_samples": {
        "en": {
            "label": "Samples",
        },
        "ar": {
            "label": "النماذج الاولية",
        },

        
    },
    "learning_rate": {
        "en": {
            "label": "Learning rate",
            "info": "Initial learning rate for AdamW.",
        },
        "ar": {
            "label": "معدل التعلم",
            "info": "معدل التعلم باستخدام AdamW",
        },

        
    },
    "num_train_epochs": {
        "en": {
            "label": "Epochs",
            "info": "Total number of training epochs to perform.",
        },
        "ar": {
            "label": "عدد الدورات",
            "info": "عدد الدورات في التدريب",
        },
 
    },
    "max_grad_norm": {
        "en": {
            "label": "Maximum gradient norm",
            "info": "Norm for gradient clipping.",
        },
        "ar": {
            "label": "الحد الاقصى لمعيار التدرج",
            "info": "معيار قص التدرج",
        },

    },
    "max_samples": {
        "en": {
            "label": "Max samples",
            "info": "Maximum samples per dataset.",
        },
        "ar": {
            "label": "الحد الا قصى للعينات",
            "info": "الحد الاقصى للعينات بعمدل البيانات",
        },

        
    },
    "compute_type": {
        "en": {
            "label": "Compute type",
            "info": "Whether to use mixed precision training.",
        },
        "ar": {
            "label": "نوع الحساب",
            "info": "ما إذا كنت تريد استخدام التدريب الدقيق المختلط",
        },
        

        
    },
    "cutoff_len": {
        "en": {
            "label": "Cutoff length",
            "info": "Max tokens in input sequence.",
        },
        "ar": {
            "label": "طول القطع",
            "info": "الحد الأقصى للرموز المميزة في تسلسل الإدخال",
        },

        
    },
    "batch_size": {
        "en": {
            "label": "Batch size",
            "info": "Number of samples processed on each GPU.",
        },
        "ar": {
            "label": "Batch size",
            "info": "عدد العينات التي تمت معالجتها على كل وحدة معالجة رسومات",
        },
        

        
    },
    "gradient_accumulation_steps": {
        "en": {
            "label": "Gradient accumulation",
            "info": "Number of steps for gradient accumulation.",
        },
        "ar": {
            "label": "تراكم التدرج",
            "info": "عدد خطوات تراكم التدرج",
        },

    },
    "val_size": {
        "en": {
            "label": "Val size",
            "info": "Proportion of data in the dev set.",
        },
        "ar": {
            "label": "Val size",
            "info": "نسبة البيانات في مجموعة التطوير",
        },

    },
    "lr_scheduler_type": {
        "en": {
            "label": "LR scheduler",
            "info": "Name of the learning rate scheduler.",
        },
        "ar": {
            "label": "LR scheduler",
            "info": "اسم جدولة معدل التعلم",
        },

        
    },
    "extra_tab": {
        "en": {
            "label": "Extra configurations",
        },
        "ar": {
            "label": "اعدادات اضافية",
        },
        

        
    },
    "logging_steps": {
        "en": {
            "label": "Logging steps",
            "info": "Number of steps between two logs.",
        },
        "ar": {
            "label": "خطوات التسجيل",
            "info": "عدد مرات الدخول",
        },

        
        
    },
    "save_steps": {
        "en": {
            "label": "Save steps",
            "info": "Number of steps between two checkpoints.",
        },
        "ar": {
            "label": "حفظ الخطوات",
            "info": "عدد الخطوات مابين ال checkpoints",
        },

        
    },
    "warmup_steps": {
        "en": {
            "label": "Warmup steps",
            "info": "Number of steps used for warmup.",
        },
        "ar": {
            "label": "خطوات الإحماء",
            "info": "عدد الخطوات المستخدمة للإحماء",
        },
        

        
    },
    "neftune_alpha": {
        "en": {
            "label": "NEFTune Alpha",
            "info": "Magnitude of noise adding to embedding vectors.",
        },
        "ar": {
            "label": "NEFTune Alpha",
            "info": "حجم الضوضاء إضافة إلى تضمين المتجهات",
        },
        
   
    },
    "optim": {
        "en": {
            "label": "Optimizer",
            "info": "The optimizer to use: adamw_torch, adamw_8bit or adafactor.",
        },
        "ar": {
            "label": "محسن",
            "info": "المحسن المراد استخدامه: adamw_torch أو adamw_8bit أو adfactor",
        },
        
    
    },
    "packing": {
        "en": {
            "label": "Pack sequences",
            "info": "Pack sequences into samples of fixed length.",
        },
        "ar": {
            "label": "تسلسل الحزمة",
            "info": "حزم التسلسلات في عينات ذات طول ثابت",
        },
        
  
        
    },
    "neat_packing": {
        "en": {
            "label": "Use neat packing",
            "info": "Avoid cross-attention between packed sequences.",
        },
        "ar": {
            "label": "استخدم التعبئة الأنيقة",
            "info": "تجنب الانتباه المتبادل بين التسلسلات المعبأة",
        },

    },
    "train_on_prompt": {
        "en": {
            "label": "Train on prompt",
            "info": "Disable the label mask on the prompt (only for SFT).",
        },
        "ar": {
            "label": "تدريب على التلقين",
            "info": "تعطيل قناع التسمية في المطالبة (فقط ل SFT)",
        },
        
        
    },
    "mask_history": {
        "en": {
            "label": "Mask history",
            "info": "Train on the last turn only (only for SFT).",
        },
        "ar": {
            "label": "تاريخ القناع",
            "info": "تدرب في المنعطف الأخير فقط (فقط ل SFT).",
        },

        
        
    },
    "resize_vocab": {
        "en": {
            "label": "Resize token embeddings",
            "info": "Resize the tokenizer vocab and the embedding layers.",
        },
        "ar": {
            "label": "تغيير حجم تضمينات الرمز المميز",
            "info": "قم بتغيير حجم مفردات الرمز المميز وطبقات التضمين",
        },
        

        
    },
    "use_llama_pro": {
        "en": {
            "label": "Enable LLaMA Pro",
            "info": "Make the parameters in the expanded blocks trainable.",
        },
        "ar": {
            "label": "تمكين لاما برو",
            "info": "اجعل المعلمات في الكتل الموسعة قابلة للتدريب",
        },
        
  
    },
    "shift_attn": {
        "en": {
            "label": "Enable S^2 Attention",
            "info": "Use shift short attention proposed by LongLoRA.",
        },
        "ar": {
            "label": "تمكين انتباه S^2",
            "info": "استخدم تحول الانتباه القصير الذي اقترحه LongLoRA.",
        },

        
    },
    "report_to": {
        "en": {
            "label": "Enable external logger",
            "info": "Use TensorBoard or wandb to log experiment.",
        },
        "ar": {
            "label": "تمكين المسجل الخارجي",
            "info": "استخدم TensorBoard أو wandb لتسجيل التجربة",
        },

    },
    "freeze_tab": {
        "en": {
            "label": "Freeze tuning configurations",
        },
        "ar": {
            "label": "تجميد تكوينات الضبط",
        },

    },
    "freeze_trainable_layers": {
        "en": {
            "label": "Trainable layers",
            "info": "Number of the last(+)/first(-) hidden layers to be set as trainable.",
        },
        "ar": {
            "label": "طبقات قابلة للتدريب",
            "info": "عدد الطبقات المخفية الأخيرة (+) / الأولى (-) ليتم تعيينها على أنها قابلة للتدريب",
        },
  
        
    },
    "freeze_trainable_modules": {
        "en": {
            "label": "Trainable modules",
            "info": "Name(s) of trainable modules. Use commas to separate multiple modules.",
        },
        "ar": {
            "label": "وحدات قابلة للتدريب",
            "info": "اسم (أسماء) الوحدات القابلة للتدريب. استخدم الفواصل لفصل وحدات متعددة",
        },

        
    },
    "freeze_extra_modules": {
        "en": {
            "label": "Extra modules (optional)",
            "info": (
                "Name(s) of modules apart from hidden layers to be set as trainable. "
                "Use commas to separate multiple modules."
            ),
        },
        "ar": {
            "label": "وحدات إضافية (اختياري)",
            "info": (
                "اسم (أسماء) الوحدات بصرف النظر عن الطبقات المخفية ليتم تعيينها على أنها قابلة للتدريب. "
                "استخدم الفواصل لفصل وحدات متعددة."
            ),
        },
        

        
    },
    "lora_tab": {
        "en": {
            "label": "LoRA configurations",
        },
        "ar": {
            "label": "اعدادت ال LoRA",
        },

        
    },
    "lora_rank": {
        "en": {
            "label": "LoRA rank",
            "info": "The rank of LoRA matrices.",
        },
        "ar": {
            "label": "تقيم ال LoRA",
            "info": "تقيم مقايس ال LoRA",
        },

        
    },
    "lora_alpha": {
        "en": {
            "label": "LoRA alpha",
            "info": "Lora scaling coefficient.",
        },
        "ar": {
            "label": "LoRA alpha",
            "info": "معامل تحجيم لورا",
        },
        

        
    },
    "lora_dropout": {
        "en": {
            "label": "LoRA dropout",
            "info": "Dropout ratio of LoRA weights.",
        },
        "ar": {
            "label": "LoRA dropout",
            "info": "نسبة التسرب من أوزان LoRA.",
        },
        
        
    },
    "loraplus_lr_ratio": {
        "en": {
            "label": "LoRA+ LR ratio",
            "info": "The LR ratio of the B matrices in LoRA.",
        },
        "ar": {
            "label": "نسبة لورا + LR ",
            "info": "نسبة LR للمصفوفات B في LoRA",
        },
        

    },
    "create_new_adapter": {
        "en": {
            "label": "Create new adapter",
            "info": "Create a new adapter with randomly initialized weight upon the existing one.",
        },
        "ar": {
            "label": "إنشاء محول جديد",
            "info": "قم بإنشاء محول جديد بوزن تمت تهيئته عشوائيا على المحول الموجود",
        },
        

    },
    "use_rslora": {
        "en": {
            "label": "Use rslora",
            "info": "Use the rank stabilization scaling factor for LoRA layer.",
        },
        "ar": {
            "label": "استخدم رسلورا",
            "info": "استخدم عامل قياس تثبيت الترتيب لطبقة LoRA.",
        },
        

    },
    "use_dora": {
        "en": {
            "label": "Use DoRA",
            "info": "Use weight-decomposed LoRA.",
        },
        "ar": {
            "label": "استخدم DoRA",
            "info": "استخدم LoRa المتحلل بالوزن",
        },
        

    },
    "use_pissa": {
        "en": {
            "label": "Use PiSSA",
            "info": "Use PiSSA method.",
        },
        "ar": {
            "label": "استخدام PiSSA",
            "info": "استخدم طريقة PiSSA.",
        },
        
        
    },
    "lora_target": {
        "en": {
            "label": "LoRA modules (optional)",
            "info": "Name(s) of modules to apply LoRA. Use commas to separate multiple modules.",
        },
        "ar": {
            "label": "وحدات LoRA (اختيارية)",
            "info": "اسم (أسماء) الوحدات لتطبيق LoRA. استخدم الفواصل لفصل وحدات متعددة.",
        },
        

        
    },
    "additional_target": {
        "en": {
            "label": "Additional modules (optional)",
            "info": (
                "Name(s) of modules apart from LoRA layers to be set as trainable. "
                "Use commas to separate multiple modules."
            ),
        },
        "ar": {
            "label": "وحدات إضافية (اختياري)",
            "info": (
                "اسم (أسماء) الوحدات بصرف النظر عن طبقات LoRA ليتم تعيينها على أنها قابلة للتدريب. "
                "استخدم الفواصل لفصل وحدات متعددة."
            ),
        },
        

        
    },
    "rlhf_tab": {
        "en": {
            "label": "RLHF configurations",
        },
        "ar": {
            "label": "اعدادات ال  RLHF",
        },

    },
    "pref_beta": {
        "en": {
            "label": "Beta value",
            "info": "Value of the beta parameter in the loss.",
        },
        "ar": {
            "label": "قيمة بيتا",
            "info": "قيمة المعلمة بيتا في الخسارة.",
        },
        

        
    },
    "pref_ftx": {
        "en": {
            "label": "Ftx gamma",
            "info": "The weight of SFT loss in the final loss.",
        },
        "ar": {
            "label": "Ftx gamma",
            "info": "وزن خسارة SFT في الخسارة النهائية.",
        },
        

    },
    "pref_loss": {
        "en": {
            "label": "Loss type",
            "info": "The type of the loss function.",
        },
        "ar": {
            "label": "نوع الخسارة",
            "info": "نوع دالة الخسارة.",
        },

        
        
    },
    "reward_model": {
        "en": {
            "label": "Reward model",
            "info": "Adapter of the reward model in PPO training.",
        },
        "ar": {
            "label": "نموذج المكافأة",
            "info": "محول نموذج المكافأة في تدريب PPO.",
        },
        

    },
    "ppo_score_norm": {
        "en": {
            "label": "Score norm",
            "info": "Normalizing scores in PPO training.",
        },
        "ar": {
            "label": "معيار النتيجة",
            "info": "تطبيع الدرجات في تدريب PPO.",
        },
        
        

    },
    "ppo_whiten_rewards": {
        "en": {
            "label": "Whiten rewards",
            "info": "Whiten the rewards in PPO training.",
        },
        "ar": {
            "label": "Whiten rewards",
            "info": "تبييض المكافآت في تدريب PPO.",
        },
        

    },
    "galore_tab": {
        "en": {
            "label": "GaLore configurations",
        },
        "ar": {
            "label": "اعدادت ال GaLore",
        },
        
        
    },
    "use_galore": {
        "en": {
            "label": "Use GaLore",
            "info": "Enable gradient low-Rank projection.",
        },
        "ar": {
            "label": "استخدم GaLore",
            "info": "تمكين الإسقاط المتدرج منخفض الرتبة.",
        },
        
        
        
    },
    "galore_rank": {
        "en": {
            "label": "GaLore rank",
            "info": "The rank of GaLore gradients.",
        },
        "ar": {
            "label": "تقيم ال GaLore",
            "info": "رتبة تدرجات GaLore.",
        },
        

        
        
    },
    "galore_update_interval": {
        "en": {
            "label": "Update interval",
            "info": "Number of steps to update the GaLore projection.",
        },
        "ar": {
            "label": "تحديث الفاصل الزمني",
            "info": "عدد الخطوات لتحديث إسقاط GaLore.",
        },
        

        
    },
    "galore_scale": {
        "en": {
            "label": "GaLore scale",
            "info": "GaLore scaling coefficient.",
        },
        "ar": {
            "label": "مقياس غالور",
            "info": "معامل تحجيم GaLore",
        },
 
        
    },
    "galore_target": {
        "en": {
            "label": "GaLore modules",
            "info": "Name(s) of modules to apply GaLore. Use commas to separate multiple modules.",
        },
        "ar": {
            "label": "وحدات GaLore",
            "info": "اسم (أسماء) الوحدات النمطية لتطبيق GaLore. استخدم الفواصل لفصل وحدات متعددة.",
        },
        
        
        
    },
    "badam_tab": {
        "en": {
            "label": "BAdam configurations",
        },
        "ar": {
            "label": "اعدادات ال BAdam",
        },
        

        
    },
    "use_badam": {
        "en": {
            "label": "Use BAdam",
            "info": "Enable the BAdam optimizer.",
        },
        "ar": {
            "label": "استخدام ال BAdam",
            "info": "قم بتمكين محسن BAdam.",
        },
        

    },
    "badam_mode": {
        "en": {
            "label": "BAdam mode",
            "info": "Whether to use layer-wise or ratio-wise BAdam optimizer.",
        },
        "ar": {
            "label": "BAdam mode",
            "info": "سواء كنت تريد استخدام محسن BAdam من حيث الطبقة أو النسبة.",
        },
        

        
    },
    "badam_switch_mode": {
        "en": {
            "label": "Switch mode",
            "info": "The strategy of picking block to update for layer-wise BAdam.",
        },
        "ar": {
            "label": "Switch mode",
            "info": "استراتيجية اختيار الكتلة لتحديثها من أجل BAdam من حيث الطبقة.",
        },
        

        
    },
    "badam_switch_interval": {
        "en": {
            "label": "Switch interval",
            "info": "Number of steps to update the block for layer-wise BAdam.",
        },
        "ar": {
            "label": "تبديل الفاصل الزمني",
            "info": "عدد الخطوات لتحديث الكتلة لطبقة BAdam",
        },
        

        
    },
    "badam_update_ratio": {
        "en": {
            "label": "Update ratio",
            "info": "The ratio of the update for ratio-wise BAdam.",
        },
        "ar": {
            "label": "نسبة التحديث",
            "info": "نسبة التحديث لنسبة BAdam.",
        },
        

        
    },
    "cmd_preview_btn": {
        "en": {
            "value": "Preview command",
        },
        "ar": {
            "value": "أمر المعاينة",
        },
        
        

        
    },
    "arg_save_btn": {
        "en": {
            "value": "Save arguments",
        },
        "ar": {
            "value": "حفظ الوسيطات",
        },
        

        
    },
    "arg_load_btn": {
        "en": {
            "value": "Load arguments",
        },
        "ar": {
            "value": "تحميل الوسيطات",
        },
        

        
    },
    "start_btn": {
        "en": {
            "value": "Start",
        },
        "ar": {
            "value": "بدء",
        },
        

    },
    "stop_btn": {
        "en": {
            "value": "Abort",
        },
        "ar": {
            "value": "أجهض",
        },

    },
    "output_dir": {
        "en": {
            "label": "Output dir",
            "info": "Directory for saving results.",
        },
        "ar": {
            "label": "مسار المخرجات",
            "info": "المسار لحفظ المخرجات",
        },
 
        
    },
    "config_path": {
        "en": {
            "label": "Config path",
            "info": "Path to config saving arguments.",
        },
        "ar": {
            "label": "مسار الضبط",
            "info": "مسار حفظ الضبط",
        },
        

    },
    "device_count": {
        "en": {
            "label": "Device count",
            "info": "Number of devices available.",
        },
        "ar": {
            "label": "عدد الأجهزة",
            "info": "عدد الأجهزة المتاحة.",
        },

    },
    "ds_stage": {
        "en": {
            "label": "DeepSpeed stage",
            "info": "DeepSpeed stage for distributed training.",
        },
        "ar": {
            "label": "مرحلة السرعة العميقة",
            "info": "مرحلة DeepSpeed للتدريب الموزع.",
        },

        
    },
    "ds_offload": {
        "en": {
            "label": "Enable offload",
            "info": "Enable DeepSpeed offload (slow down training).",
        },
        "ar": {
            "label": "تمكين إلغاء التحميل",
            "info": "تمكين إلغاء تحميل DeepSpeed (إبطاء التدريب).",
        },
        

    },
    "output_box": {
        "en": {
            "value": "Ready.",
        },
        "ar": {
            "value": "جاهز.",
        },

    },
    "loss_viewer": {
        "en": {
            "label": "Loss",
        },
        "ar": {
            "label": "الخسارة",
        },

        
    },
    "predict": {
        "en": {
            "label": "Save predictions",
        },
        "ar": {
            "label": "حفظ التوقعات",
        },

        
    },
    "infer_backend": {
        "en": {
            "label": "Inference engine",
        },
        "ar": {
            "label": "محرك الاستدلال",
        },

        
    },
    "infer_dtype": {
        "en": {
            "label": "Inference data type",
        },
        "ar": {
            "label": "نوع بيانات الاستدلال",
        },

        
    },
    "load_btn": {
        "en": {
            "value": "Load model",
        },
        "ar": {
            "value": "تحميل النموذج",
        },
        

        
    },
    "unload_btn": {
        "en": {
            "value": "Unload model",
        },
        "ar": {
            "value": "عدم تحميل المودل",
        },
        

    },
    "info_box": {
        "en": {
            "value": "Model unloaded, please load a model first.",
        },
        "ar": {
            "value": "تم تفريغ النموذج ، يرجى تحميل النموذج أولا.",
        },
        

        
    },
    "role": {
        "en": {
            "label": "Role",
        },
        "ar": {
            "label": "الدور",
        },
        
 
        
    },
    "system": {
        "en": {
            "placeholder": "System prompt (optional)",
        },
        "ar": {
            "placeholder": "موجه النظام (اختياري)",
        },
        

        
    },
    "tools": {
        "en": {
            "placeholder": "Tools (optional)",
        },
        "ar": {
            "placeholder": "الأدوات (اختياري)",
        },
        

        
    },
    "image": {
        "en": {
            "label": "Image (optional)",
        },
        "ar": {
            "label": "صورة (اختياري)",
        },
        

    },
    "video": {
        "en": {
            "label": "Video (optional)",
        },
        "ar": {
            "label": "فيديو (اختياري)",
        },

        
        
    },
    "query": {
        "en": {
            "placeholder": "Input...",
        },
        "ar": {
            "placeholder": "الادخال...",
        },
        

        
    },
    "submit_btn": {
        "en": {
            "value": "Submit",
        },
        "ar": {
            "value": "إرسال",
        },
        

    },
    "max_length": {
        "en": {
            "label": "Maximum length",
        },
        "ar": {
            "label": "الحد الأقصى للطول",
        },
        

        
    },
    "max_new_tokens": {
        "en": {
            "label": "Maximum new tokens",
        },
        "ar": {
            "label": "الحد الأقصى للرموز الجديدة",
        },
        

        
    },
    "top_p": {
        "en": {
            "label": "Top-p",
        },
        "ar": {
            "label": "Top-p",
        },
        

        
    },
    "temperature": {
        "en": {
            "label": "Temperature",
        },
        "ar": {
            "label": "درجة الحرارة",
        },
        

    },
    "clear_btn": {
        "en": {
            "value": "Clear history",
        },
        "ar": {
            "value": "مسح التاريخ",
        },
        

    },
    "export_size": {
        "en": {
            "label": "Max shard size (GB)",
            "info": "The maximum size for a model file.",
        },
        "ar": {
            "label": "الحد الأقصى لحجم القشرة (جيجابايت)",
            "info": "الحد الأقصى لحجم ملف النموذج.",
        },
        

        
        
    },
    "export_quantization_bit": {
        "en": {
            "label": "Export quantization bit.",
            "info": "Quantizing the exported model.",
        },
        "ar": {
            "label": "تصدير بت الكمية.",
            "info": "قياس النموذج المصدر.",
        },
        
        
    },
    "export_quantization_dataset": {
        "en": {
            "label": "Export quantization dataset",
            "info": "The calibration dataset used for quantization.",
        },
        "ar": {
            "label": "تصدير مجموعة بيانات التكميم",
            "info": "مجموعة بيانات المعايرة المستخدمة في التكميم.",
        },
        
        
    },
    "export_device": {
        "en": {
            "label": "Export device",
            "info": "Which device should be used to export model.",
        },
        "ar": {
            "label": "تصدير الجهاز",
            "info": "الجهاز الذي يجب استخدامه لتصدير النموذج.",
        },
        
 
    },
    "export_legacy_format": {
        "en": {
            "label": "Export legacy format",
            "info": "Do not use safetensors to save the model.",
        },
        "ar": {
            "label": "تصدير التنسيق القديم",
            "info": "لا تستخدم أجهزة الأمان لحفظ النموذج.",
        },
        
        
    },
    "export_dir": {
        "en": {
            "label": "Export dir",
            "info": "Directory to save exported model.",
        },
        "ar": {
            "label": "مسار الاخراج",
            "info": "دليل لحفظ النموذج المصدر.",
        },
        
        
    },
    "export_hub_model_id": {
        "en": {
            "label": "HF Hub ID (optional)",
            "info": "Repo ID for uploading model to Hugging Face hub.",
        },
        "ar": {
            "label": "HF Hub ID (optional)",
            "info": "معرف الريبو لتحميل النموذج إلى مركز معانقة الوجه.",
        },
                
        
    },
    "export_btn": {
        "en": {
            "value": "Export",
        },
        "ar": {
            "value": "اصدار",
        },
          
        
    },
}


ALERTS = {
    "err_conflict": {
        "en": "A process is in running, please abort it first.",
        "ar": "هناك عملية قيد التشغيل ، يرجى إجهاضها أولا.",
        
        
    },
    "err_exists": {
        "en": "You have loaded a model, please unload it first.",
        "ar": "لقد قمت بتحميل نموذج ، يرجى تفريغه أولا",
        
        
    },
    "err_no_model": {
        "en": "Please select a model.",
        "ar": "الرجاء اختيار نموذج.",
        
        
    },
    "err_no_path": {
        "en": "Model not found.",
        "ar": "لم يتم العثور على النموذج.",
        
        
    },
    "err_no_dataset": {
        "en": "Please choose a dataset.",
        "ar": "الرجاء اختيار مجموعة بيانات.",
        
        
    },
    "err_no_adapter": {
        "en": "Please select an adapter.",
        "ar": "الرجاء تحديد محول.",
        
        
    },
    "err_no_output_dir": {
        "en": "Please provide output dir.",
        "ar": "يرجي تقديم مسار المخرجات",
        
        
    },
    "err_no_reward_model": {
        "en": "Please select a reward model.",
        "ar": "يرجى تحديد نموذج المكافأة.",
        
        
    },
    "err_no_export_dir": {
        "en": "Please provide export dir.",
        "ar": "يرجي تحديد مسار الاصدار",
        
        
    },
    "err_gptq_lora": {
        "en": "Please merge adapters before quantizing the model.",
        "ar": "يرجى دمج المحولات قبل تحديد حجم النموذج.",
        
        
    },
    "err_failed": {
        "en": "Failed.",
        "ar": "فشل",
        
        
    },
    "err_demo": {
        "en": "Training is unavailable in demo mode, duplicate the space to a private one first.",
        "ar": "التدريب غير متوفر في الوضع التجريبي ، قم بتكرار المساحة إلى مساحة خاصة أولا.",
        
        
    },
    "err_tool_name": {
        "en": "Tool name not found.",
        "ar": "لم يتم العثور على اسم الأداة.",
        
        
    },
    "err_json_schema": {
        "en": "Invalid JSON schema.",
        "ar": "مخطط JSON غير صالح.",
        
        
    },
    "err_config_not_found": {
        "en": "Config file is not found.",
        "ar": "لم يتم العثور على ملف الاعدادات",
        
        
    },
    "warn_no_cuda": {
        "en": "CUDA environment was not detected.",
        "ar": "لم يتم الكشف عن بيئة CUDA.",
        
        
    },
    "warn_output_dir_exists": {
        "en": "Output dir already exists, will resume training from here.",
        "ar": "مسار المخرجات موجود مسبقا, سوف نستأنف التدريب من هنا ",
        
        
    },
    "info_aborting": {
        "en": "Aborted, wait for terminating...",
        "ar": "تم إحباطه ، انتظر الإنهاء ...",
        
        
    },
    "info_aborted": {
        "en": "Ready.",
        "ar": "جاهز.",
        
        
    },
    "info_finished": {
        "en": "Finished.",
        "ar": "منته",
        
        
    },
    "info_config_saved": {
        "en": "Arguments have been saved at: ",
        "ar": "تم حفظ الوسيطات في: ",
        
        
    },
    "info_config_loaded": {
        "en": "Arguments have been restored.",
        "ar": "تمت استعادة الحجج.",
        
        
    },
    "info_loading": {
        "en": "Loading model...",
        "ar": "النموذج قيد التحميل",
        
        
    },
    "info_unloading": {
        "en": "Unloading model...",
        "ar": "يتم رفع النموذج",
        
        
    },
    "info_loaded": {
        "en": "Model loaded, now you can chat with your model!",
        "ar": "تم تحميل النموذج , الان يمكنك استخدام النموذج!",
        
        
    },
    "info_unloaded": {
        "en": "Model unloaded.",
        "ar": "تم تحميل النموذج",
        
        
    },
    "info_exporting": {
        "en": "Exporting model...",
        "ar": "قيد الاصدار",
        
        
    },
    "info_exported": {
        "en": "Model exported.",
        "ar": "تم اصدار النموذج",
        
        
    },
}
