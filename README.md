# Beethoven Piano Sonata with Function Harmony (BPS-FH) dataset

The BPS-FH dataset consists of the symbolic musical data and functional harmony annotations of the 1st movements from Beethoven's 32 piano sonatas. The dataset can be used for functional harmony recognition, and also for symbolic cohrd estimation. More details can be found in the paper: 

[Tsung-Ping Chen and Li Su, “Functional Harmony Recognition with Multi-task Recurrent Neural Networks,”  International Society of Music Information Retrieval Conference (ISMIR), September 2018.](http://ismir2018.ircam.fr/doc/pdfs/178_Paper.pdf)



Every piece contins 5 labels, including 1) note events, 2) beats, 3) donw beats, 4) chords, and 5) phrases.

**Notes Events**

The columns represent onset (in crotchet beats), MIDI note number, morphetic pitch number, duration (in crotchet beats), staff number (integers from zero for the top staff), and measure (-1 for incomplete measure).

**Beats and Down Beats**

The ontime of beats and down beats.

**Chords**

The columns represent onset, offset, key, degree, quality, inversion, Roman numeral notation.

--

**Musical Form**

For users who are interested in human analyses of musical form, you are recommended to convert the annotations in the BPS-FH dataset by the parser provided in https://github.com/MarkGotham/Taking-Form.

A converted example is provided in the folder "Taking Form."
