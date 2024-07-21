package io.github.aqmeraamir;

import org.bukkit.event.Listener;
import org.bukkit.plugin.java.JavaPlugin;

import java.util.Objects;

public class NeuralCraft extends JavaPlugin {

    @Override
    public void onEnable() {
        // Initialization prompt
        getLogger().info("Plugin is enabled!");

        // Set up the event listener
        Listener blockPlaceListener = new EventListener();
        getServer().getPluginManager().registerEvents(blockPlaceListener, this);

        // Setup executors for commands
        Objects.requireNonNull(this.getCommand("digit")).setExecutor(new PredictDigit(blockPlaceListener));

    }

    @Override
    public void onDisable() {
        getLogger().info("Plugin is disabled!");
    }
}
