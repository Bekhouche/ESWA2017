function face = face_alignment(IMG,face_offset,face_size,Eyes)
% 2D face alignement
%
%       face = face_alignment(IMG,face_offset,face_size,Eyes)
% 
%             Input:
%               IMG : Image of size [M x N x K] 
%               face_offset : [K_side K_top K_bottom]
%               face_size :  [M' x N']
%               Eyes : [Right_Eye_X Left_Eye_X; Right_Eye_Y Left_Eye_Y]
%
%             Output:
%               face - Image of size [M' x N' x K] 
% 
% Examples:
%
%       img = imread('kids.tif');
%       face_offset = [0.5 1 1.75];
%       face_size = [224 224];
%       eyes = [125 156; 100 95];
%       face = face_alignment(img,face_offset,face_size,eyes);
%       imshow(face);
% 
% Reference:
%	[1] BEKHOUCHE, S. E. (2017). Facial Soft Biometrics: Extracting demographic traits (Doctoral dissertation, Faculté des sciences et technologies).
%   [2] Bekhouche, Salah Eddine, et al. "Pyramid multi-level features for facial demographic estimation." Expert Systems with Applications 80 (2017): 297-310.
%
%   Written by Salah Eddine Bekhouche (salah AT bekhouche.com)
%

tg_a = diff(Eyes(2,:))/diff(Eyes(1,:));
angle = tg_a*(180/pi);
%tg_a = -angle * (pi/180);
IMG_R = imrotate(IMG, angle);
Cx = size(IMG,2)/2;
Cy = size(IMG,1)/2;
Ex = (size(IMG_R,2) - size(IMG,2))/2;
Ey = (size(IMG_R,1) - size(IMG,1))/2;
R_EyeX = (Cx+(Eyes(1,1)-Cx)*cos(tg_a)-(Eyes(2,1)-Cy)*sin(tg_a))+Ex;
R_EyeY = (Cy+(Eyes(1,1)-Cx)*sin(tg_a)+(Eyes(2,1)-Cy)*cos(tg_a))+Ey;
L_EyeX = (Cx+(Eyes(1,2)-Cx)*cos(tg_a)-(Eyes(2,2)-Cy)*sin(tg_a))+Ex;
%L_EyeY = (Cy+(Eyes(1,2)-Cx)*sin(tg_a)+(Eyes(2,2)-Cy)*cos(tg_a))+Ey;

d = sqrt((L_EyeX - R_EyeX)^2);
K_side = face_offset(1);
K_top = face_offset(2);
K_bottom = face_offset(3);
face = imcrop(IMG_R, [R_EyeX-(K_side*d) R_EyeY-(K_top*d) d+2*(K_side*d) (K_top*d)+(K_bottom*d)]);
face = imresize(face,face_size);
end