<script setup>
import { computed } from 'vue'
import {
  darkTheme,
  NConfigProvider,
  NMessageProvider,
  NDialogProvider,
} from 'naive-ui'
import { useThemeStore } from './stores/theme'
import { buildOverrides } from './theme'

const themeStore = useThemeStore()
themeStore.init()

const naiveTheme = computed(() => (themeStore.isDark ? darkTheme : null))
const themeOverrides = computed(() => buildOverrides(themeStore.activePalette, themeStore.isDark))
</script>

<template>
  <NConfigProvider :theme="naiveTheme" :theme-overrides="themeOverrides" class="app-root">
    <NMessageProvider class="app-root">
      <NDialogProvider class="app-root">
        <router-view />
      </NDialogProvider>
    </NMessageProvider>
  </NConfigProvider>
</template>

<style>
/* Naive UI 的 Provider 组件包裹在 #app 与真实视图之间，
   必须让这条链都成为 flex column 并撑满高度，否则 .app-shell 的 flex:1 不生效
   （flex 上下文不会被 Provider 自动继承） */
.app-root {
  display: flex;
  flex-direction: column;
  flex: 1 1 0;
  min-height: 0;
}
</style>
