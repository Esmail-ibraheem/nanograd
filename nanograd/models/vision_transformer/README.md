
run the file either using the terminal :
```Bash
python inference.py --model_path "%USERPROFILE%\projects\paligemma-weights\paligemma-3b-pt-224" --prompt "this building is " --image_file_path "test_images\pic1.jpeg" --max_tokens_to_generate 100 --temperature 0.8 --top_p 0.9 --do_sample False --only_cpu True
```

or run using the shell script:
```shell
#!/bin/bash

MODEL_PATH="$HOME/projects/paligemma-weights/paligemma-3b-pt-224"
PROMPT="this building is "
IMAGE_FILE_PATH="test_images/pic1.jpeg"
MAX_TOKENS_TO_GENERATE=100
TEMPERATURE=0.8
TOP_P=0.9
DO_SAMPLE="False"
ONLY_CPU="True"

python inference.py \
    --model_path "$MODEL_PATH" \
    --prompt "$PROMPT" \
    --image_file_path "$IMAGE_FILE_PATH" \
    --max_tokens_to_generate $MAX_TOKENS_TO_GENERATE \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --do_sample $DO_SAMPLE \
    --only_cpu $ONLY_CPU \


```

**by running this `./launch_inference.sh`**
