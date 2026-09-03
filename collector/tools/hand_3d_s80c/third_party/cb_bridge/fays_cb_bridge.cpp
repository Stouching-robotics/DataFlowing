// fays_cb_bridge.cpp — ViKit 双目帧回调桥接（C ABI，ctypes 可调）。
//
// 背景：worker 用 FAYS_VIK_GetStereoFrames 轮询取帧时，SDK 内部
// 装配线程写帧缓冲与该函数的内拷贝竞态，会在交付帧内留下水平缝
//（缝行随机漂移、两目共模、settle-poll 已证返回后我方缓冲稳定——
// 即缝在 SDK 内部拷贝瞬间已冻结进帧）。官方 stereo_depth_gui 改用
// FAYS_VIK_RegisterStereoImageCallback 回调取帧（SDK 装配完成后才
// 回调，帧必完整）且无撕裂。
//
// 该注册 API 形参是 C++ std::function，ctypes 无法直调；本桥接在
// C++ 侧包一层：回调把帧 memmove 进自持 4 槽环形缓冲（seqlock 发布），
// Python 侧 fays_cb_poll 取最新完整帧。回调运行在 SDK 装配线程，
// 槽内 memmove ~6MB ≈ 1ms，与官方 GUI 回调内 copyTo+cvtColor 同量级。
//
// 不链接 libfays_vikit.so：worker 已 RTLD_GLOBAL 加载 vikit，本桥接
// dlsym(RTLD_DEFAULT) 解析注册函数——SDK 版本无关，编译一次通用。
// 依赖仅 vendor 头 fays_atrak_types.h（AtrakImage 布局）与 libdl。

#include "fays_atrak/fays_atrak_types.h"

#include <atomic>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <functional>

namespace {

constexpr int kRingSlots = 4;

struct Slot {
    uint8_t* data = nullptr;
    size_t   bytes = 0;      // 本帧实际字节数（已按槽容量截断）
    int      width = 0, height = 0, channel = 0, step = 0, seq = 0;
    short    encoding = 0;
    ull      timestamp = 0;
};

Slot*   g_slots = nullptr;
size_t  g_slot_bytes = 0;
std::atomic<uint64_t> g_write_seq{0};   // 回调线程：已完整写入帧数
std::atomic<bool>     g_stopped{false};
uint64_t g_read_seq = 0;                // Python 线程：已读帧数

void stereo_cb(AtrakImage* img)
{
    if (g_stopped.load(std::memory_order_acquire) || !g_slots
        || !img || !img->data || img->bytes == 0)
        return;
    uint64_t n = g_write_seq.load(std::memory_order_relaxed);
    Slot& s = g_slots[n % kRingSlots];
    size_t nb = img->bytes;
    if (nb > g_slot_bytes)
        nb = g_slot_bytes;              // 防御：超槽容量截断（不应发生）
    std::memmove(s.data, img->data, nb);
    s.bytes = nb;
    s.width = img->width;
    s.height = img->height;
    s.channel = img->channel;
    s.step = img->step;
    s.encoding = img->encoding;
    s.seq = img->seq;
    s.timestamp = img->timestamp;
    // 元数据先落，seq 最后 release 发布（poll 侧 acquire 后见全槽）
    g_write_seq.store(n + 1, std::memory_order_release);
}

using RegisterFn = int (*)(void*, std::function<void(AtrakImage*)>);

}  // namespace

extern "C" {

// 注册回调并分配环。handle=CreateHandleWithConfig 返回的句柄；
// slot_bytes=单帧字节上限（worker 传 MAX_STEREO_BYTES）。返回 0 成功。
int fays_cb_register(void* handle, size_t slot_bytes)
{
    if (!handle || slot_bytes == 0)
        return -1;
    // Itanium ABI 固定修饰名（与 SDK 头文件签名逐字对应；worker 已
    // RTLD_GLOBAL 加载 vikit，RTLD_DEFAULT 可解析）
    auto* fn = reinterpret_cast<RegisterFn>(dlsym(
        RTLD_DEFAULT,
        "_Z36FAYS_VIK_RegisterStereoImageCallbackPvSt8functionIFvP10AtrakImageEE"));
    if (!fn)
        return -2;                      // 符号缺失：SDK 版本不含回调 API

    g_slots = new Slot[kRingSlots];
    g_slot_bytes = slot_bytes;
    for (int i = 0; i < kRingSlots; ++i)
        g_slots[i].data = new uint8_t[slot_bytes];
    g_stopped.store(false);
    g_write_seq.store(0);
    g_read_seq = 0;

    return fn(handle, std::function<void(AtrakImage*)>(stereo_cb));
}

// 取最新完整帧到 dst（容量 dst_n）。有新帧返回 1 并填元数据；
// 无新帧返回 0；未注册/空 dst 返回 -1。seqlock：拷贝后重读写序号，
// 若拷贝窗口内被写入线程绕环（>=kRingSlots 帧）则重试一次
//（50fps + ~1ms 拷贝下几乎不可能，纯防御）。
int fays_cb_poll(void* dst, size_t dst_n,
                 unsigned long long* ts_ns,
                 int* w, int* h, int* ch, short* encoding,
                 int* step, unsigned int* bytes, int* seq)
{
    if (!g_slots || !dst)
        return -1;
    for (int attempt = 0; attempt < 2; ++attempt) {
        uint64_t n = g_write_seq.load(std::memory_order_acquire);
        if (n == 0 || n == g_read_seq)
            return 0;
        const Slot& s = g_slots[(n - 1) % kRingSlots];
        size_t nb = s.bytes < dst_n ? s.bytes : dst_n;
        std::memmove(dst, s.data, nb);
        if (g_write_seq.load(std::memory_order_acquire) - (n - 1)
            >= kRingSlots)
            continue;                   // 拷贝窗口内被绕环：重试
        g_read_seq = n;
        *ts_ns = s.timestamp;
        *w = s.width;
        *h = s.height;
        *ch = s.channel;
        *encoding = s.encoding;
        *step = s.step;
        *bytes = static_cast<unsigned int>(nb);
        *seq = s.seq;
        return 1;
    }
    return 0;
}

// 停止回调写入（worker 在 DestroyHandle 前调用，防销毁竞态期
// SDK 残留回调写环）
void fays_cb_stop()
{
    g_stopped.store(true, std::memory_order_release);
}

}  // extern "C"
