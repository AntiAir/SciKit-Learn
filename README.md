Test

    # -*- coding: utf-8 -*-
    ##第一行一定要先加 (有關中文編碼) .
    ##Alt+3 Alt+4 快速加註 減註
    #
    from sklearn import datasets
    from sklearn.linear_model import LinearRegression
    import matplotlib.pyplot as plt
    
    ##loaded_data = datasets.load_boston()
    ##data_X = loaded_data.data
    ##data_y = loaded_data.target
    ##
    ##model = LinearRegression()
    ##model.fit(data_X , data_y)
    ##
    ##print(model.predict(data_X[:4, :]))     #產生預測值
    ##print(data_y[:4])                                       #實際值
    
    X, y = datasets.make_regression (n_samples = 100, n_features =1, n_targets =1, noise =1)
    plt.scatter(X,y)
    plt.show()

![Graph](https://github.com/AntiAir/SciKit-Learn/blob/master/regression.png?raw=true)

I revised it.

# Big Word

1. Survey
2. Review
3. Conclusion

![GG](d:\GitHub\SciKit-Learn\regression.png)