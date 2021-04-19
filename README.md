# SmartCar-laneDetection
[![State-of-the-art Shitcode](https://img.shields.io/static/v1?label=State-of-the-art&message=Shitcode&color=7B5804)](https://github.com/trekhleb/state-of-the-art-shitcode)

智能车赛道边缘检测与处理

这个分支的大概思路是：通过对假定为白色赛道的像素灰度进行累计移动平均的方法获得白色参考值，并通过将当前像素与参考值做差得到的值的大小是否超过阈值来判断是否是黑色。

这种方法可能可行，但不确定。