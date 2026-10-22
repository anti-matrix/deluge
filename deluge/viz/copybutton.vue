<template>
  <button @click="copy()" :title="title"
          style="height:30px;padding:0 10px;background:#fff;color:#111;border:1px solid #888;border-radius:3px;cursor:pointer">
    {{ done ? done_label : label }}
  </button>
</template>
<script>
module.exports = {
  data() { return {done: false}; },
  methods: {
    copy() {
      const text = this.text || '';
      const fallback = () => {  // no clipboard API (an http page): a hidden text area
        const t = document.createElement('textarea');
        t.value = text; document.body.appendChild(t); t.select();
        try { document.execCommand('copy'); } finally { document.body.removeChild(t); }
      };
      if (navigator.clipboard && window.isSecureContext) navigator.clipboard.writeText(text).catch(fallback);
      else fallback();
      this.done = true;
      setTimeout(() => { this.done = false; }, 1500);
    },
  },
};
</script>
