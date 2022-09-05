## Eye written characters preprocessing

[paper link] (https://www.mdpi.com/2079-9292/10/23/3009 "paper link")

> - Donghyun Kang
> 
> We conducted three pre-processing steps - Eye blink removal, normalization, and data resampling.
> This code having the data normalizaion and data reampling without the eye blink removal methods.
> The data resampling process is including 'Removing data points with closed position -> Fill points -> Distance re-adjust -> Equalization & Minimize points -> Distance > re-adjust'

![preprocessing](https://user-images.githubusercontent.com/48580174/137443414-b34f5afb-d677-44f3-b6da-d8a98243805a.png)


## Eye written characters classification

> We chose a algorithm which called Siamese Network to classify these characters.
> In addition, We using new characters named 'Reference data'. These dataset became the benchmark of each class
