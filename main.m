clear all;
LFDir = [fileparts(mfilename('fullpath')) '/'];

%% Parameters
LFName = 'buddha';
uRange = [1:9];		% list of u indices of the images to load from the files.
vRange = [1:9];		% list of v indices of the images to load from the files.
crop = [0 0 0 0];		% croped pixels form the input images [left,right,top,bottom].
nU = length(uRange);
nV = length(vRange);
outer_scale = 1.91;
inner_scale = 0.08;
angle_max = 0.7;
angle_min = -5*pi/12;

%% Load Light Field data
tic;fprintf('Load 4D Light Field...');
LF = loadLF([LFDir LFName], '', 'bmp', uRange, vRange, crop);
t=toc;fprintf(['\b\b\b (done in ' num2str(t) 's)\n']);

nY = size(LF, 1);
nX = size(LF, 2);
nC = size(LF, 3);

vC = ceil((nV+1)/2);
uC = ceil((nU+1)/2);
im_ref = LF(:,:,:,vC,uC);

%% Sturcture Tensor for Depth Estimation
depth_array = zeros(1,nU,nY,nX,'double');
for epi_h = 1:nY
    fprintf(['Processing %d/',num2str(nY),'\n'],epi_h);
    
    % epipolar plane image (EPI) extraction in center view row (v=vC)
    epi_horizontal = zeros(nU,nX,nC,'uint8');
    for u = 1 : nU
        epi_horizontal(u,:,:) = LF(epi_h,:,:,vC,u);
    end
    epi_horizontal = im2double(rgb2gray(epi_horizontal));
    
    % Gaussian filter on EPI
    G_outer = fspecial('gaussian', [4*ceil(outer_scale)+1 4*ceil(outer_scale)+1], outer_scale);
    Ic = imfilter(epi_horizontal,G_outer,'symmetric','same');
    
    % compute gradient of EPI and enhancement
    [Sx,Sy] = gradient(Ic);
    Sx = convolution_column(size(epi_horizontal,1), 3.0/16.0, 10.0/16.0, 3.0/16.0, Sx);
    Sy = convolution_row(size(epi_horizontal,2), 3.0/16.0, 10.0/16.0, 3.0/16.0, Sy);
    
    % construct the structure tensor
    G_inner = fspecial('gaussian', [4*ceil(inner_scale)+1 4*ceil(inner_scale)+1], inner_scale);
    Jxx = imfilter(Sx.*Sx,G_inner,'symmetric','same');
    Jyy = imfilter(Sy.*Sy,G_inner,'symmetric','same');
    Jxy = imfilter(Sx.*Sy,G_inner,'symmetric','same');
    
    % estimate the depth
    angle = atan2(Jyy-Jxx,2*Jxy)/2;
    angle(angle>angle_max)=angle_max;
    angle(angle<angle_min)=angle_min;
    depth_epi = tan(angle);
%     depth_epi(depth_epi>0.1)=0.1;
%     depth_epi(depth_epi<-1.73)=-1.73; 
    depth_array(1,:,epi_h,:) = depth_epi;
end

depth_array = mat2gray(depth_array);
for u = 1:nU
    depth_view = depth_array(1,u,:,:);
    depth_view = squeeze(depth_view);
   
    imshow(depth_view);
%     imwrite(depth_view,[LFName,'_depth','_',num2str(vC),'_',num2str(u),'.png']);
end


function out = convolution_column(H, k0, k1, k2, in)

if H > 1
   g(1,:) = (k2 * in(2,:) + k1 * in(1,:)) / (k1 + k2);
   g(H,:) = (k1 * in(H,:) - k0 * in(H-1,:)) / (k0 + k1);
end

% Take centered differences on interior points
if H > 2
   g(2:H-1,:) = k2 * in(3:H,:) + k0 * in(1:H-2,:) + k1 * in(2:H-1,:);
end

out = g;

end

function out = convolution_row(W, k0, k1, k2, in)

if W > 1
   g(:,1) = (k2 * in(:,2) + k1 * in(:,1)) / (k1 + k2);
   g(:,W) = (k1 * in(:,W) - k0 * in(:,W-1)) / (k0 + k1);
end

% Take centered differences on interior points
if W > 2
   g(:,2:W-1) = k2 * in(:,3:W) + k0 * in(:,1:W-2) + k1 * in(:,2:W-1);
end

out = g;

end
