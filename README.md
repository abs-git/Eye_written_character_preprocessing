## Eye written characters preprocessing

paper link : https://www.mdpi.com/2079-9292/10/23/3009

> - Donghyun Kang
> 
> We conducted three pre-processing steps - Eye blink removal, normalization, and data resampling. <br>
> 
> In the version 1, our preprocessing pipeline has only two steps as a data normalizaion and data reampling using dataset after eye blink removing. <br>
> The pipeline constructed the 5 stages with 8 functions following 'Data normalization -> Removing closed points -> Fill points -> Distance re-adjust -> Points Equalization & Minimization -> Distance re-adjust' <br>
>
> In the version 2, we refactored these pipeline. <br>
> The new pipeline contains 4 functinos following 'Data normalization -> Removing closed points -> Fill points & Points Minimization -> Distance re-adjust' <br>
> And, we added more datasets for the robust experiments as a 'without eye blink removing characters' and 'katakana characters' <br>
>
> As the result, we could save the time for preprocessing from 40.7s to 15.1s using the 'eye blink removed characters'

![preprocessing](https://user-images.githubusercontent.com/48580174/137443414-b34f5afb-d677-44f3-b6da-d8a98243805a.png)


## Eye written characters classification

> We chose a algorithm which called Siamese Network to classify these characters.
> In addition, We using new characters named 'Reference data'. These dataset became the benchmark of each class
