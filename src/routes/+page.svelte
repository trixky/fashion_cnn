<!-- ================================================= SCRIPT -->
<script lang="ts">
  import { browser } from "$app/environment";
  import { onMount, tick } from "svelte";
  import train from "$lib/logic/train";
  import evaluate from "$lib/logic/evaluate";
  import * as tf from "@tensorflow/tfjs";

  interface Dataset {
    inputs: number[][];
    outputs: number[];
  }

  let dataset: Dataset | null = null;
  let model: tf.Sequential | null | undefined = undefined;
  let epoch = -1;
  let accuracy = 0;
  let trainingStarted = false;
  let inputEvaluation: number[] = new Array(28 * 28).fill(0);
  let answerEvaluation: number | null = null;
  let expectedEvaluation: number | null = null;

  function downloadDataset() {
    console.log("download dataset...");
    fetch("./mnist.json", {
      method: "GET",
      headers: {
        Accept: "application/json",
      },
    })
      .then((response) => response.json())
      .then((response) => {
        dataset = response as Dataset;
        epoch = -1;
        accuracy = 0;
      });
  }

  function trainCallBack(newEpoch: number, newAccuracy: number) {
    epoch = newEpoch;
    accuracy = newAccuracy;
  }

  async function handleTrain() {
    if (dataset !== null) {
      trainingStarted = true;
      epoch = -1;
      accuracy = 0;
      await tick();
      setTimeout(async () => {
        if (dataset !== null)
          model = await train(dataset.inputs, dataset.outputs, trainCallBack);
      }, 50);
    }
  }

  function handleEvaluate() {
    if (dataset !== null && model !== null && model !== undefined) {
      const { answer, expected, input } = evaluate(
        model,
        dataset.inputs,
        dataset.outputs
      );

      inputEvaluation = input;
      answerEvaluation = answer;
      expectedEvaluation = expected;
    }
  }

  onMount(() => {
    if (browser) downloadDataset();
  });
</script>

<!-- ================================================= CONTENT -->
<div id="form" class="flex flex-col justify-center items-center gap-3 w-full">
  {#if dataset !== null}
    <p>dataset loaded</p>
  {:else}
    <p>dataset loading...</p>
  {/if}
  <button on:click={handleTrain} disabled={dataset === null || trainingStarted}
    >train model</button
  >
  <div id="progress" class="my-2">
    <div>
      <p>epoch:</p>
      <p>&nbsp;{(epoch + 1).toString().padStart(2, " ")} / 50</p>
    </div>
    <div>
      <p>accuracy:</p>
      <p>
        {Math.round(accuracy * 100)
          .toString()
          .padStart(3, " ")} %
      </p>
    </div>
  </div>
  <button
    on:click={handleEvaluate}
    disabled={dataset === null || model === null || model === undefined}
    >prediction</button
  >
</div>
<ol class="mt-5">
  {#each inputEvaluation as cell}
    {@const intensity = Math.min(Math.floor(cell * 10), 9)}
    <li class="bg-black bg-opacity-80 w-2 h-2 md:w-3 md:h-3" style="opacity: {cell + 0.1};">
      <p class="text-[0.7em] md:text-xs">{intensity}</p>
    </li>
  {/each}
</ol>
<div id="result" class="flex justify-between w-full mt-3">
  <p>
    <span class:nothing={model === null || model === undefined}>expected:</span
    ><span class="nothing">&nbsp;</span><span class:nothing={expectedEvaluation === null}
      >{expectedEvaluation === null ? "?" : expectedEvaluation}</span
    >
  </p>
  <p>
    <span class:nothing={model === null || model === undefined}>answer:</span
    ><span class="nothing">&nbsp;</span><span
      class:nothing={answerEvaluation === null}
      class:success={answerEvaluation !== null &&
        expectedEvaluation === answerEvaluation}
      class:failure={answerEvaluation !== null &&
        expectedEvaluation !== answerEvaluation}
    >
      {answerEvaluation === null ? "?" : answerEvaluation}
    </span>
  </p>
</div>
<div id="about" class="text-xs opacity-50 italic text-left w-full mt-5">
  <div>
    <p>dataset:</p>
    <p>10 000 images (28x28)</p>
  </div>
  <div>
    <p>validation split:</p>
    <p>20% (8 000)</p>
  </div>
  <div>
    <p>hidden layers (2):</p>
    <p>32 + 16</p>
  </div>
  <div>
    <p>activation functions:</p>
    <p>relu & softmax</p>
  </div>
</div>

<!-- ================================================= CSS -->
<style lang="postcss">
  #progress > div {
    @apply flex items-center justify-center;
  }

  #progress > div > p {
    @apply w-32 text-right;
  }

  #progress > div > p:last-child {
    @apply text-left;
  }

  #form p {
    white-space: pre;
  }

  ol {
    user-select: none;
    display: grid;
    grid-template-columns: repeat(28, 1fr);
    gap: 1px;
  }

  #result {
    @apply transition-opacity duration-200;
  }

  .success {
    @apply text-green-300;
  }

  .failure {
    @apply text-red-300;
  }

  .nothing {
    @apply opacity-10;
  }

  #about {
    @apply flex flex-col gap-[2px];
  }

  #about > div {
    @apply flex justify-between;
  }
</style>
