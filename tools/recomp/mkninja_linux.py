"""Write build.ninja for the Linux x86-64 build (Steam Deck target): generated TUs + runtime + NVN/Vulkan backend.

usage: python3 mkninja_linux.py [--null] [--root DIR] [--gen DIR] [--out DIR] [--obj DIR] [--deps DIR] [--syslib DIR]

  --null   null GPU backend (nvn/backend_null.c: presents nothing, no Vulkan / shaderc): headless smoke tests
  --root   source tree (default: two levels above this script)
  --gen    generated C++ (default: ROOT/gen/main; gen.py --prefix s output is ROOT/gen/sdk, next to it)
  --out    build directory (build.ninja, runtime + backend objects, the totk binary); default /root/totk-linux/build[-null]
  --obj    generated-TU objects, shared by the Vulkan and null builds (default /root/totk-linux/obj)
  --deps   third-party install prefix with sdl3-install/, opus-install/ (default /root/totk-linux/deps)
  --gen-opt OPT       compile the generated TUs with OPT (e.g. -O1) instead of -O2 (runtime / backend objects stay -O2)
  --gen-keep FILE     objects (paths relative to --obj, e.g. gen/tu_000.o) that keep the -O2 rule: already built, so ninja does not redo them
  --gen-pool N        generated TUs compiled at the same time (default 2; ~1.3 GB each)
  --ram-guard SCRIPT  run before every generated TU compile (e.g. wait until the Windows host has free RAM)
  --prebuilt-gen      do not build the generated objects here, only link them (they were built by another build dir sharing --obj)
  --syslib directory with libvpx.a and libshaderc_combined.a (default: the first of /usr/lib/x86_64-linux-gnu, /usr/lib64, /usr/lib that has libvpx.a)
  --pgo gen           instrumented build (-fprofile-generate=pgo, -DTOTK_PGO_GEN): rt_main writes pgo/*.profraw (relative to the
                      run directory) at TOTK_PGO_DUMP seconds
  --pgo use:FILE      build with that merged profile (llvm-profdata-19 merge -o FILE pgo/*.profraw); -mllvm -pgso=false as on Windows
  env CC / CXX: the clang to use (default clang-19 / clang++-19; clang >= 17 with lld works)

Mirrors tools/recomp/mkninja.py (Windows, untouched): same source sets, clang -O2 -march=x86-64-v3 (+ -mtune=znver2 for the Zen 2 Deck),
-fno-exceptions. Generated TUs are the same commands in both builds so their objects are shared.
Build:  nice -n 19 ninja -C <out> -j2 objs-gen   (the 188 + 28 generated TUs, big: at most 2 at a time)
        nice -n 19 ninja -C <out> -j4            (everything else + link)
"""
import glob, os, sys

args = sys.argv[1:]
def opt(name, default=None):
    if name in args:
        i = args.index(name)
        v = args[i + 1]
        del args[i:i + 2]
        return v
    return default
null = '--null' in args
if null: args.remove('--null')
prebuilt = '--prebuilt-gen' in args
if prebuilt: args.remove('--prebuilt-gen')
pgo = opt('--pgo', '')                              # gen | use:<profdata> (Deck PGO, 2026-10-16: the Windows PGO exe was +12-18 %)
gen_opt = opt('--gen-opt')                          # e.g. -O1: optimisation level of the generated TUs (default: the -O2 of cxxflags)
gen_pool = int(opt('--gen-pool', '2'))              # generated TUs compiled at the same time (ninja pool depth)
gen_keep = opt('--gen-keep')                        # file: objects (path relative to --obj) that keep the default -O2 rule
guard = opt('--ram-guard')                          # script run before every generated TU (waits for free host RAM)
here = os.path.dirname(os.path.abspath(__file__))
root = os.path.abspath(opt('--root', os.path.join(here, '..', '..')))
gen = os.path.abspath(opt('--gen', os.path.join(root, 'gen', 'main')))
out = os.path.abspath(opt('--out', '/root/totk-linux/build-null' if null else '/root/totk-linux/build'))
obj = os.path.abspath(opt('--obj', '/root/totk-linux/obj'))
deps = os.path.abspath(opt('--deps', '/root/totk-linux/deps'))
RT = os.path.join(root, 'runtime')
SDL = os.path.join(deps, 'sdl3-install')
OPUS = os.path.join(deps, 'opus-install')
CC, CXX = os.environ.get('CC', 'clang-19'), os.environ.get('CXX', 'clang++-19')

def esc(p):                       # ninja path escaping
    return p.replace('$', '$$').replace(' ', '$ ').replace(':', '$:')

rt_cpp = sorted(glob.glob(os.path.join(RT, '*.cpp'))) + [os.path.join(RT, 'linux', 'plat_posix.cpp')]
nvn_c = sorted(glob.glob(os.path.join(RT, 'nvn', '*.c')))
vk_c, mw_c = [], []
if null:
    # all of nvn/*.c including backend_null.c; the vulkan/ and maxwell/ trees are not built. runtime/linux/null_stubs.c supplies the
    # few symbols the runtime takes from the Vulkan backend (totk_hooks.c / exec.c / device.c)
    nvn_c.append(os.path.join(RT, 'linux', 'null_stubs.c'))
else:
    nvn_c = [s for s in nvn_c if os.path.basename(s) != 'backend_null.c']
    vk_c = [s for s in sorted(glob.glob(os.path.join(RT, 'nvn', 'vulkan', '*.c'))) if os.path.basename(s) != 'shim_win.c']
    vk_c.append(os.path.join(RT, 'linux', 'shim_sdl3.c'))
    mw_c = sorted(glob.glob(os.path.join(RT, 'nvn', 'maxwell', '*.c')))

os.makedirs(out, exist_ok=True)
sdk_dir = os.path.join(os.path.dirname(gen), 'sdk')
inc = f'-I{RT}/linux -I{RT}'
cxxflags = f'-std=c++20 -O2 -march=x86-64-v3 -mtune=znver2 -fno-exceptions -fno-pic -Wno-everything -pthread -I{SDL}/include {inc}'
cflags = f'-std=gnu17 -O2 -march=x86-64-v3 -mtune=znver2 -fno-pic -Wno-everything -pthread -I{SDL}/include -I{RT}/linux -I{RT}/nvn'
# Vulkan headers newer than the distro's (VK_KHR_unified_image_layouts, feedback-loop info: 2026-10-19 runtime): a copy of the
# Vulkan SDK's include/ at deps/vulkan-headers (1.4.357 on 2026-10-20); everything new is checked with has_ext at run time.
VKH = os.path.join(deps, 'vulkan-headers', 'include')   # tools/linux/build_deps.sh fetches Vulkan-Headers 1.4.357 there
if not null and not os.path.isdir(VKH): sys.exit(f'missing {VKH}: run tools/linux/build_deps.sh {deps}')
cvkflags = f'$cflags' + (f' -I{VKH}' if os.path.isdir(VKH) else '') + f' -I{RT}/linux/vmashim -include {RT}/linux/compat_posix.h'
pgo_cc = pgo_link = ''
if pgo == 'gen':
    pgo_cc, pgo_link = ' -fprofile-generate=pgo -DTOTK_PGO_GEN', ' -fprofile-generate=pgo'
elif pgo.startswith('use:'):
    pgo_cc = f' -fprofile-use={pgo[4:]} -mllvm -pgso=false'
elif pgo:
    sys.exit(f'--pgo {pgo}: gen or use:<profdata>')
cxxflags += pgo_cc
cflags += pgo_cc

with open(os.path.join(out, 'build.ninja'), 'w') as f:
    f.write(f'cxx = {CXX}\ncc_ = {CC}\ncxxflags = {cxxflags}\ncflags = {cflags}\ncvkflags = {cvkflags}\n')
    f.write(f'pool gen\n  depth = {gen_pool}\n')              # generated TUs: ~1.3 GB of RAM each (up to ~2 GB at -O2)
    f.write('rule cxx\n  command = $cxx $cxxflags -MD -MF $out.d -c $in -o $out\n  depfile = $out.d\n  deps = gcc\n  description = CXX $out\n')
    # clang 22 crashed in RegisterCoalescer on some generated functions with SLP + AVX2 (Windows build, e.g. f_1f3e5c): retry such a TU
    # without SLP so only that TU loses it
    f.write("rule gen\n  command = sh -c '$cxx $cxxflags -MD -MF $out.d -c $in -o $out || $cxx $cxxflags -fno-slp-vectorize -MD -MF $out.d -c $in -o $out'\n"
            '  depfile = $out.d\n  deps = gcc\n  description = GEN $out\n  pool = gen\n')
    keep = set(l.strip() for l in open(gen_keep)) if gen_keep and os.path.exists(gen_keep) else set()
    if gen_opt:
        # same rule at another optimisation level (-O1 compiles ~3x faster than -O2: a linkable binary in hours, not a day). The
        # guard script runs first, before every TU, and may wait (host RAM); `;` so a missing guard never blocks the build.
        f1 = cxxflags.replace(' -O2 ', f' {gen_opt} ')
        pre = f'{guard}; ' if guard else ''
        f.write(f"rule gen1\n  command = sh -c '{pre}$cxx {f1} -MD -MF $out.d -c $in -o $out || $cxx {f1} -fno-slp-vectorize -MD -MF $out.d -c $in -o $out'\n"
                '  depfile = $out.d\n  deps = gcc\n  description = GEN1 $out\n  pool = gen\n')
    f.write('rule c\n  command = $cc_ $cflags -MD -MF $out.d -c $in -o $out\n  depfile = $out.d\n  deps = gcc\n  description = CC $out\n')
    f.write('rule cvk\n  command = $cc_ $cvkflags -MD -MF $out.d -c $in -o $out\n  depfile = $out.d\n  deps = gcc\n  description = CVK $out\n')
    gen_objs, objs = [], []
    def gen_edge(sub, src):
        o = os.path.join(obj, sub, os.path.splitext(os.path.basename(src))[0] + '.o')
        if not prebuilt:       # (--prebuilt-gen: the objects already exist, built by the other build dir; this one only links them)
            rule = 'gen1' if gen_opt and f'{sub}/{os.path.basename(o)}' not in keep else 'gen'
            f.write(f'build {esc(o)}: {rule} {esc(src)}\n')
        gen_objs.append(o)
    for src in sorted(glob.glob(os.path.join(gen, '*.cpp'))): gen_edge('gen', src)
    for src in sorted(glob.glob(os.path.join(sdk_dir, '*.cpp'))): gen_edge('sdk', src)
    for src in rt_cpp:
        o = os.path.join(out, 'rt', os.path.splitext(os.path.basename(src))[0] + '.o')
        f.write(f'build {esc(o)}: cxx {esc(src)}\n'); objs.append(o)
    for src in nvn_c:
        o = os.path.join(out, 'nvn', os.path.splitext(os.path.basename(src))[0] + '.o')
        f.write(f'build {esc(o)}: c {esc(src)}\n'); objs.append(o)
    for src in vk_c + mw_c:
        sub = os.path.basename(os.path.dirname(src))
        o = os.path.join(out, 'nvn', sub, os.path.splitext(os.path.basename(src))[0] + '.o')
        f.write(f'build {esc(o)}: cvk {esc(src)}\n'); objs.append(o)
    f.write('build objs-gen: phony ' + ' '.join(esc(o) for o in gen_objs) + '\n')
    f.write('build objs-rt: phony ' + ' '.join(esc(o) for o in objs) + '\n')
    # static: shaderc (combined, Ubuntu libshaderc-dev), libvpx, libopus 1.5.2; shared: SDL3 (bundled as lib/libSDL3.so.0), libvulkan
    # (the system loader), glibc. libstdc++ / libgcc are linked statically so the binary needs nothing newer than glibc 2.39.
    sys_a = opt('--syslib') or next((d for d in ('/usr/lib/x86_64-linux-gnu', '/usr/lib64', '/usr/lib') if os.path.exists(d + '/libvpx.a')), '/usr/lib/x86_64-linux-gnu')
    libs = f'-L{SDL}/lib -lSDL3 {OPUS}/lib/libopus.a {sys_a}/libvpx.a'
    if not null:
        # (Ubuntu's libshaderc_combined.a is a thin 250 KB archive: glslang and SPIRV-Tools come from the glslang-dev and spirv-tools
        # packages; a shaderc built from google/shaderc already contains them and the extra archives then resolve nothing)
        libs += f' -lvulkan -Wl,--start-group {sys_a}/libshaderc_combined.a'
        for extra in ('libglslang.a', 'libglslang-default-resource-limits.a', 'libSPIRV.a', 'libSPIRV-Tools-opt.a', 'libSPIRV-Tools.a'):
            if os.path.exists(os.path.join(sys_a, extra)): libs += f' {sys_a}/{extra}'
        libs += ' -Wl,--end-group'
    libs += ' -ldl -lpthread -lm'
    f.write('rule link\n  command = $cxx' + pgo_link + ' -fuse-ld=lld -no-pie -static-libstdc++ -static-libgcc -pthread -Wl,-z,noexecstack -Wl,-Map=$out.map '
            "-Wl,-rpath,'$$ORIGIN/lib' -o $out @$out.rsp " + libs + '\n'
            '  rspfile = $out.rsp\n  rspfile_content = $in\n  description = LINK $out\n')
    exe = os.path.join(out, 'totk')
    f.write(f'build {esc(exe)}: link ' + ' '.join(esc(o) for o in gen_objs + objs) + '\n')
    f.write(f'default {esc(exe)}\n')
print('backend:', 'null' if null else 'vulkan', '|', len(gen_objs), 'generated objects,', len(objs), 'runtime objects ->', out)
