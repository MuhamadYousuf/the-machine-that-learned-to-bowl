import {loadFont as loadMontserrat} from '@remotion/google-fonts/Montserrat';
import {loadFont as loadMono} from '@remotion/google-fonts/JetBrainsMono';

export const display = loadMontserrat('normal', {weights: ['600', '800', '900'], subsets: ['latin']}).fontFamily;
export const mono = loadMono('normal', {weights: ['500', '700'], subsets: ['latin']}).fontFamily;
