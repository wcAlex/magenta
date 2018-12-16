# Tuning RNNs with RL


## Code structure
*   In the constructor, RLTuner loads the `q_network`, `target_q_network`, and
    `reward_rnn` from a checkpointed Note RNN.

*   The tensorflow graph architecture is defined in the `build_graph`
    function.

*   The model is trained using the `train` function. It will continuously
    place notes by calling `action`, receive rewards using `collect_reward`,
    and save these experiences using `store`.

*   The network weights are updated using `training_step`, which samples
    minibatches of experience from the model's `experience` buffer and uses
    this to compute gradients based on the loss function in `build_graph`.

*   During training, the function `evaluate_model` is occasionally run to
    test how much reward the model receives from both the Reward RNN and the
    music theory functions.

*   After the model is trained, you can use the `save_model_and_figs` function
    to save a checkpoint of the model and a set of figures of the rewards over
    time.

*   Finally, use `generate_music_sequence` to generate a melody with your
    trained model! You can also call this function before training, to see how
    the model's songs have improved with training! If you set the
    `visualize_probs` parameter to *True*, it will also plot the
    note probabilities of the model over time.

## Running the code

1. Set up development environment
```angular2html
	1. Create python 3 virtual env and active the env
	2. pip install magenta
	3. git clone https://github.com/tensorflow/magenta.git
	4. python setup.py develop
```

2. Train RL Tuner and generate music
```angular2html
python rl_tuner_train.py  \
--note_rnn_checkpoint_dir={ LSTM model checkpoint directory } \
--note_rnn_checkpoint_file={ LSTM model checkpoint file name } \
--note_rnn_type={LSTM Model Type} \
--note_rnn_hparams="batch_size=64,rnn_layer_sizes=[64,64]" \
--output_dir={Output dir for generated music, checkpoint file and evaluation graph} \
--algorithm=q \
--reward_scaler=0.1 \
--output_every_nth=50000 \
--num_notes_in_melody=128 \
--exploration_mode=boltzmann \
--midi_primer="{Prime for mimi generation}"
```

Example:
```angular2html
python rl_tuner_train.py --note_rnn_checkpoint_dir=/tmp/magenta/melody_rnn/attention_rnn/logdir/run1/train/ \ 
--note_rnn_checkpoint_file=model.ckpt --note_rnn_type=attention_rnn --training_steps=100000 \ 
--exploration_steps=50000 --output_every_nth=5000  --output_dir=/tmp/magenta/melody_rnn/rl_tuner/attention_rnn \
--algorithm=q --reward_scaler=0.1 --output_every_nth=50000 --num_notes_in_melody=128 \
 --exploration_mode=boltzmann --midi_primer="[60, -2, 60, -2, 67, -2, 67, -2]"
```
You will find generated music and graphs in /tmp/magenta/melody_rnn/rl_tuner/attention_rnn for above example.

## Improving the model
If you have ideas for improving the sound of the model based on your own rules
for musical aesthetics, try modifying the `reward_music_theory` function!

## Helpful links

*   The original code implementation of the model described in [this paper][our arxiv].
*   Code repo is https://github.com/wcAlex/magenta
*   For more on DQN, see [this paper][dqn].
*   The DQN code was originally based on [this example][dqn ex].

[our arxiv]: https://arxiv.org/pdf/1611.02796v2.pdf
[blog post]: https://magenta.tensorflow.org/2016/11/09/tuning-recurrent-networks-with-reinforcement-learning/
[ipynb]: https://nbviewer.jupyter.org/github/tensorflow/magenta/tree/master/magenta/models/rl_tuner/rl_tuner.ipynb
[note rnn ckpt]: http://download.magenta.tensorflow.org/models/rl_tuner_note_rnn.ckpt
[magenta pretrained]: https://github.com/tensorflow/magenta/tree/master/magenta/models/melody_rnn#pre-trained
[dqn ex]: https://github.com/nivwusquorum/tensorflow-deepq/blob/master/tf_rl/
[g learning]: https://arxiv.org/pdf/1512.08562.pdf
[psi learning]: http://homepages.inf.ed.ac.uk/svijayak/publications/rawlik-RSS2012.pdf
[dqn]: https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
