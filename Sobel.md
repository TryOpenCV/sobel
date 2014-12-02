### Sobel ###

***

#### Теория ####
Оператор Собеля используется в области обработки изображений. Часто его применяют в алгоритмах выделения границ. По сути, это дискретный дифференциальный оператор, вычисляющий приближенное значение градиента яркости изображения. Результатом применения оператора Собеля в каждой точке изображения является либо вектор градиента яркости в этой точке, либо его норма. Оператор Собеля основан на свёртке изображения небольшими сепарабельными целочисленными фильтрами в вертикальном и горизонтальном направлениях, поэтому его относительно легко вычислять.

Математически, градиент функции двух переменных для каждой точки изображения (которой и является функция яркости) — двумерный вектор, компонентами которого являются производные яркости изображения по горизонтали и вертикали. В каждой точке изображения градиентный вектор ориентирован в направлении наибольшего увеличения яркости, а его длина соответствует величине изменения яркости. Это означает, что результатом оператора Собеля в точке области постоянной яркости будет нулевой вектор, а в точке, лежащей на границе областей различной яркости — вектор, пересекающий границу в направлении увеличения яркости.

Рассмотрим пример:

Пусть мы хотим найти некаторые границы на изображении.

![](http://docs.opencv.org/_images/Sobel_Derivatives_Tutorial_Theory_0.jpg)

Высокое изменение градиента указывает на значительные изменения в изображении.
Для более наглядного примера посчитаем, что у нас есть 1D-изображение.

![](http://docs.opencv.org/_images/Sobel_Derivatives_Tutorial_Theory_Intensity_Function.jpg)

Более отчетливо можно увидет и определить изменения, если мы возьмем первую производную:

![](http://docs.opencv.org/_images/Sobel_Derivatives_Tutorial_Theory_dIntensity_Function.jpg)

Таким образом можно сделать вывод, что для определения границы изображения нужно выбирать тот пиксель, у которого наиболее высокий градиент по сравнению с соседними.

#### Формализация ####
Оператор собеля использует ядра 3×3, с которыми сворачивают исходное изображение для вычисления приближенных значений производных по горизонтали и по вертикали. Пусть A исходное изображение, а Gx и Gy — два изображения, где каждая точка содержит приближенные производные по x и по y. Они вычисляются следующим образом:

![](https://upload.wikimedia.org/math/9/8/a/98a9514f83f3e81531216e92efb212a3.png)
(* обозначает двумерную операцию свертки)

В каждой точке изображения приближенное значение величины градиента можно вычислить, используя полученные приближенные значения производных:

![](https://upload.wikimedia.org/math/4/9/4/4941373cd3ab2fe1b807a06e879c323b.png)

Иногда для простоты используется:

![](http://docs.opencv.org/_images/math/a938de1258efc69f677cc4ff9d6c430b3530580a.png)

#### Пример кода ####
Применяет оператор Собеля и выдает на выходе изображение с обнаруженными краями (яркие пиксели на более темном фоне).
```c++
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>

using namespace cv;

int main( int argc, char** argv )
{

  Mat src, src_gray;
  Mat grad;
  char* window_name = "Sobel Demo - Simple Edge Detector";
  int scale = 1;
  int delta = 0;
  int ddepth = CV_16S;

  int c;

  /// Загружаем изображение
  src = imread( argv[1] );

  if( !src.data )
  { return -1; }

  GaussianBlur( src, src, Size(3,3), 0, 0, BORDER_DEFAULT );

  /// Конвертируем его
  cvtColor( src, src_gray, CV_RGB2GRAY );

  /// Создаем новое окно
  namedWindow( window_name, CV_WINDOW_AUTOSIZE );

  /// Объявляем градиенты grad_x и grad_y
  Mat grad_x, grad_y;
  Mat abs_grad_x, abs_grad_y;

  /// Градиент X
  //Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
  Sobel( src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
  convertScaleAbs( grad_x, abs_grad_x );

  /// Градиент Y
  //Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
  Sobel( src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
  convertScaleAbs( grad_y, abs_grad_y );

  /// Слияние градиентов (аппроксимация)
  addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );

  imshow( window_name, grad );

  waitKey(0);

  return 0;
  }
```
#### Поясниния ####
Объявляем переменные:
```c++
Mat src, src_gray;
Mat grad;
char* window_name = "Sobel Demo - Simple Edge Detector";
int scale = 1;
int delta = 0;
int ddepth = CV_16S;
```
Загружаем изображение:
```c++
src = imread( argv[1] );

if( !src.data )
{ return -1; }
```
Сначала применяем функцию [GaussianBlur](http://docs.opencv.org/modules/imgproc/doc/filtering.html?highlight=gaussianblur#gaussianblur) к нашему изображению для уменьшения шума (размер ядра 3x3):
```c++
GaussianBlur( src, src, Size(3,3), 0, 0, BORDER_DEFAULT );
```
Конвертируем изображение в формат 'оттенки серого':
```c++
cvtColor( src, src_gray, CV_RGB2GRAY );
```
Вычисляем "производные" по горизонтальному и вертикальному направлению (по х и у). Для этого используем функцию Собеля:
```c++
Mat grad_x, grad_y;
Mat abs_grad_x, abs_grad_y;

/// Градиент по X
Sobel( src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
/// Градиент по Y
Sobel( src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
```
```
C++: void Sobel(InputArray src, OutputArray dst, int ddepth, int dx, int dy, int ksize=3, double scale=1, double delta=0, int borderType=BORDER_DEFAULT )
```
**Аргументы функции:**
* **src** - входное изображение.
* **dst** - изображение на выходе (такой же разимер и кол-во каналов как и у входного изображения).
* **ddepth** - [глубина цвета](https://ru.wikipedia.org/wiki/%D0%93%D0%BB%D1%83%D0%B1%D0%B8%D0%BD%D0%B0_%D1%86%D0%B2%D0%B5%D1%82%D0%B0)  (изображения). Может быть определена как src.depth():
    * src.depth() = CV_8U, ddepth = -1/CV_16S/CV_32F/CV_64F
    * src.depth() = CV_8U, ddepth = -1/CV_16S/CV_32F/CV_64F
    * src.depth() = CV_8U, ddepth = -1/CV_16S/CV_32F/CV_64F
* **dx** - пороядок производной по X
* **dy** - пороядок производной по Y
* **ksize** - размер ядра свертки (может быть равным: 1, 3, 5 или 7)
* **scale** - опционально масштабный коэффициент для вычисленных производных величин; по умолчанию, масштабирование не применяется (см [getDerivKernels ()](http://docs.opencv.org/modules/imgproc/doc/filtering.html?highlight=sobel#void getDerivKernels(OutputArray kx, OutputArray ky, int dx, int dy, int ksize, bool normalize, int ktype)) для более подробной информации).
* **delta** - необязательно дельта значение, которое добавляется к результату перед сохранением его в **dst**.
* **borderType** - задает метод экстрополяции пикселей на изображении (см [borderInterpolate()](http://docs.opencv.org/modules/imgproc/doc/filtering.html?highlight=sobel#int borderInterpolate(int p, int len, int borderType)) для более подробной информации).

Конвертируем наш промежуточный результат обратно в CV_8U (тюк при вызове функции Sobel мы указали CV_16S формат, чтобы избежать переполнения):
```c++
convertScaleAbs( grad_x, abs_grad_x );
convertScaleAbs( grad_y, abs_grad_y );
```
Аппроксимируем изображение:
```c++
addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );
```
Выводим изображение:
```c++
imshow( window_name, grad );
```
#### Результат ####
![](http://docs.opencv.org/_images/Sobel_Derivatives_Tutorial_Result.jpg)