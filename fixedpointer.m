N = 256;
angle = 2*pi*(0:(N-1))/N;

s = sin(angle)';
thd_ref_1 = ssinthd(s,1,N,1,'direct' );
thd_ref_2p5 = ssinthd(s,5/2,2*N,5,'linear' );

cs = cordicsin(angle,50)';
thd_ref_1c = ssinthd(cs,1,N,1,'direct' );
thd_ref_2p5c = ssinthd(cs,5/2,2*N,5,'linear' );

open_system('sldemo_tonegen');
set_param('sldemo_tonegen', 'StopFcn','');
out = sim('sldemo_tonegen');

currentFig = figure('Color',[1,1,1]);
subplot(3,1,1), plot(out.tonegenOut.time, out.tonegenOut.signals(1).values); grid
title({'Difference between direct lookup', 'and reference signal'});
subplot(3,1,2), plot(out.tonegenOut.time, out.tonegenOut.signals(2).values); grid
title({'Difference between interpolated lookup', 'and reference signal'});
subplot(3,1,3), plot(out.tonegenOut.time, out.tonegenOut.signals(3).values); grid
title({'Difference between CORDIC sine', 'and reference signal'});