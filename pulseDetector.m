clear; clc; close all;

rng('default');
PulseLen = 64;
theta = rand(PulseLen, 1);
pulse = exp(1i*2*pi*theta);