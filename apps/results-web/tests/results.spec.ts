import { expect, test } from "@playwright/test";

test("publishes runs and compares two executions", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "Benchmark runs" })).toBeVisible();
  await expect(page.locator("tbody tr")).toHaveCount(6);
  const compare = page.getByRole("button", { name: /Compare/ });
  await page.getByRole("checkbox").nth(0).check();
  await page.getByRole("checkbox").nth(1).check();
  await expect(compare).toBeEnabled();
  await compare.click();
  await expect(page).toHaveURL(/\/compare\//);
  await expect(page.getByRole("heading", { name: "Task comparison" })).toBeVisible();
  await expect(page.locator("tbody tr")).toHaveCount(2);
});

test("opens a stable run route with provenance", async ({ page }) => {
  await page.goto("/runs/run-a14413528f18257c7fba67c7");
  await expect(page.getByRole("heading", { name: /Qwen3.6-35B-A3B-GGUF/ })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Provenance" })).toBeVisible();
  await expect(page.getByText("osolmaz/benchmark-run-index")).toBeVisible();
  await expect(page.getByText("Public metadata only")).toBeVisible();
});
