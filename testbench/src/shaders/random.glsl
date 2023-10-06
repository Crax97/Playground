
// those infamous oneliners
float random(float co) { return fract(sin(co*(91.3458)) * 47453.5453); }
float random(vec2 co){ return fract(sin(dot(co.xy ,vec2(12.9898,78.233))) * 43758.5453); }
float random(vec3 co){ return random(co.xy+random(co.z)); }

