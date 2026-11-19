import {Config} from '@remotion/cli/config';

// GPU WebGL through ANGLE so three.js renders on the RTX 3060.
Config.setChromiumOpenGlRenderer('angle');
Config.setVideoImageFormat('png');
Config.setCodec('h264');
Config.setCrf(16);
Config.setPixelFormat('yuv420p');
Config.setConcurrency(4);
