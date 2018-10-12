# hello 1
## hello 2
### hello 3
#### hello 4
##### hello 5
###### hello 6
So when write blog, using ## as the article title, the # size is the website title.


无法获得锁 /var/lib/dpkg/lock - open (11: 资源临时不可用)

解决：
其实这是因为有另外一个程序在运行，导致锁不可用。原因可能是上次运行更新或安装没有正常完成。解决办法是杀死此进程
sudo rm /var/cache/apt/archives/lock
sudo rm /var/lib/dpkg/lock
`$Log(\theta)$`

Example (1): $h_\theta(x) = \Large\frac{1}{1 + \mathcal{e}^{(-\theta^\top x)}}$ ; example (2): $a^2 + b^2 = c^2$ ; example (3): $\sum_{i=1}^m y^{(i)}$
