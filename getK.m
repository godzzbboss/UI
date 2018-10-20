function [ kxx_, kx_x_ ] = getK(data_train_x_path, data_test_x_path)
% ����˾���
% kxx_:k(x,x')
% kx_x_:k(x',x')
% disp('hello')
data_train_x = cell2mat(struct2cell(load(data_train_x_path)));
data_test_x = cell2mat(struct2cell(load(data_test_x_path)));
covFunc = {@covADD,{[1,2,3],@covSEiso}};
hyp.cov = [-1.842740292268587;-1.832711854588193;-1.885982913342324;0.694276506764406;2.168288412418211;...
            3.205004692342746;-6.757540793156383;0.321641327954946;4.086047335463364;1.068431046439123;0.000485644205509];
kxx_ = feval(covFunc{:}, hyp.cov, data_train_x, data_test_x);
kx_x_ = feval(covFunc{:}, hyp.cov, data_test_x, data_test_x);
end


