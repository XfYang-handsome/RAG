<script setup>
// 棱镜 Logo：使用 public/prism-logo.png（透明底彩色棱镜）
// color 属性保留仅为兼容旧调用，图片模式下不生效
defineProps({
  size: { type: Number, default: 24 },
  color: { type: String, default: 'currentColor' },
})
</script>

<template>
  <img
    src="/prism-logo.png"
    :height="size"
    alt="PrismRAG"
    class="prism-logo"
    aria-hidden="true"
  />
</template>

<style scoped>
.prism-logo {
  display: block;
  width: auto;
  object-fit: contain;
  /* 白底图在深色顶栏上以圆角图标形式呈现 */
  border-radius: 22%;
}
</style>
