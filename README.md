# gst-dataset
Data Preparation for GPT-SoVITS

## Download signal youtube videos
Collect all the oral speech video from public source(Youtube). The signal videos should contain only single speaker with duration more than 10 mins.

Place the youtube link (e.g. https://www.youtube.com/watch?v=xxxxxxxxxx) into a individual txt file. One video link per line. The file name should with prefix "_signal.txt"

```
./download_signal.sh {my_dataset}
```

It will automatically download all the files into "input_{my_dataset}" directory.

## Clean and prepare the dataset.
One click script to perform all the processing in the orders:
1. Remove background noise/music.
2. Enhance speech quality with demucs.
3. Remove silence clip more than 200ms.
4. Recognize the text from audio (ASR) with whisper. The recognization process will split the audio into sentence segments.
5. Filter audio segements within 5-10s. (Alignment with the original GSV training data)
6. Extract the SSL, Bert, Hubert informations according to GSV training dataset structure.

```
python prepare_dataset.py --exp_name {my_dataset} --input_dir input_{my_dataset}
```

## Train with GSV stock webui
The exp name is directly the {my_dataset}
