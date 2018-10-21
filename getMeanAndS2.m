function [mean, s2] = getMeanAndS2(data_train_x_path,data_train_y_path,data_test_x_path)

data_train_x = cell2mat(struct2cell(load(data_train_x_path)));
data_train_y = cell2mat(struct2cell(load(data_train_y_path)));
data_test_x = cell2mat(struct2cell(load(data_test_x_path)));
hyp.mean = 0.019752085671582;
hyp.cov = [-1.842740292268587;-1.832711854588193;-1.885982913342324;0.694276506764406;2.168288412418211;...
            3.205004692342746;-6.757540793156383;0.321641327954946;4.086047335463364;1.068431046439123;0.000485644205509];
hyp.lik = -0.410976659690484;
infFunc = {@infGaussLik};
meanFunc = {@meanConst};
covFunc = {@covADD,{[1,2,3],@covSEiso}};
likFunc = {@likGauss};
[mean, s2] = gp(hyp,infFunc, meanFunc, covFunc, likFunc, data_train_x, data_train_y, data_test_x);
end

