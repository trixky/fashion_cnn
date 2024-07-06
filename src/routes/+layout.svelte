<!-- ================================================= SCRIPT -->
<script lang="ts">
  import "../styles/reset.css";
  import "../styles/tailwind.css";
  import "../styles/main.css";

  import Header from "$lib/components/layout/header.svelte";
  import Footer from "$lib/components/layout/footer.svelte";
</script>

<!-- ================================================= CONTENT -->
<div
  class="relative mx-auto flex flex-col items-center w-full h-full min-h-[100vh] overflow-hidden"
>
  <div
    class="relative mx-auto flex flex-col items-center w-full md:w-[500px] bg-black h-full min-h-[100vh] bg-opacity-90 overflow-hidden"
  >
    <div
      class="flex lg:flex justify-center items-center absolute -z-10 h-[100vh] w-[100vw] min-w-[140vh] bg-black"
    >
      <img src="/background.gif" alt="background" />
    </div>
    <Header />
    <main class="flex flex-col justify-center items-center p-4">
      <slot />
    </main>
    <Footer />
  </div>
</div>

<!-- ================================================= CSS -->
<style lang="postcss">
  /* make animation the image need to make a infinit 360 rotation */
  img {
    animation: spin 60s linear infinite;
    transform: scale(1.2);

    -webkit-filter: grayscale(100%); /* Safari 6.0 - 9.0 */
    filter: grayscale(100%);
  }

  @keyframes spin {
    0% {
      transform: scale(1.2) rotate(0deg);
    }
    100% {
      transform: scale(1.2) rotate(360deg);
    }
  }
</style>
